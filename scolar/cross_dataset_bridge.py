"""Prepare aligned reference and target datasets for cross-dataset scOLAR evaluation."""

import numpy as np
import scipy.sparse
import scanpy as sc
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder


def prepare_cross_dataset(args, ontology_manager):
    """Load one aligned AnnData file and return loaders and metadata matching the main training contract."""
    h5ad_path = getattr(args, "cross_h5ad", "data/ALIGNED_Mus_musculus_Mammary_Gland.h5ad")
    ref_name = getattr(args, "cross_ref_name", "Quake_Smart-seq2_Mammary_Gland")
    tgt_name = getattr(args, "cross_tgt_name", "Quake_10x_Mammary_Gland")
    ds_col = getattr(args, "cross_dataset_col", "dataset_name")
    label_col = getattr(args, "cross_label_col", "cell_ontology_class")

    print(f"\n=== [CrossData] {ref_name} (ref) -> {tgt_name} (tgt) ===")
    print(f"=== [CrossData] file={h5ad_path}, label={label_col} ===")

    adata = sc.read_h5ad(h5ad_path)
    if ds_col not in adata.obs:
        raise KeyError(f"[CrossData] dataset column '{ds_col}' not in obs.")
    keep = adata.obs[ds_col].astype(str).isin([ref_name, tgt_name]).values
    adata = adata[keep].copy()
    print(
        f"[CrossData] Subset to {adata.n_obs} cells "
        f"({(adata.obs[ds_col].astype(str) == ref_name).sum()} ref, "
        f"{(adata.obs[ds_col].astype(str) == tgt_name).sum()} tgt)."
    )

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.hvg, subset=False)
    hv_mask = adata.var["highly_variable"].values

    X_full = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    raw_full = adata.layers["counts"]
    raw_full = raw_full.toarray() if scipy.sparse.issparse(raw_full) else np.array(raw_full)

    sf_full = raw_full.sum(axis=1, keepdims=True).astype(np.float32)
    sf_full = np.maximum(sf_full, 1.0)
    sf = (sf_full / (np.median(sf_full) + 1e-8)).astype(np.float32)
    sf = np.clip(sf, 0.1, 10.0)

    X = X_full[:, hv_mask].astype(np.float32)
    raw_X = raw_full[:, hv_mask].astype(np.float32)

    labels = adata.obs[label_col].astype(str).values
    ds_origin = adata.obs[ds_col].astype(str).values
    is_ref = ds_origin == ref_name
    is_tgt = ds_origin == tgt_name

    le = LabelEncoder()
    raw_label_ids = le.fit_transform(labels)

    standardization = getattr(args, "cross_standardization", "joint")
    if standardization == "joint":
        # Primary manuscript protocol: fit one transform on the ref+tgt union.
        # This is transductive but label-free.
        mean_val = X.mean(axis=0, dtype=np.float64)
        std_val = X.std(axis=0, dtype=np.float64)
        std_val[std_val < 1e-8] = 1.0
        X = ((X - mean_val) / std_val).astype(np.float32)
        X = np.clip(X, -10.0, 10.0)
        print("[CrossData] Joint gene-wise z-score on ref+tgt union; clip [-10, 10].")
    elif standardization == "per_dataset":
        # Explicit sensitivity mode. This uses target-distribution statistics and
        # must not be mixed into the primary cross-dataset comparison.
        print("[CrossData] SENSITIVITY MODE: per-dataset gene-wise z-score.")
        for ds_name, mask in [(ref_name, is_ref), (tgt_name, is_tgt)]:
            if mask.sum() == 0:
                continue
            X_sub = X[mask]
            mean_val = X_sub.mean(axis=0, dtype=np.float64)
            std_val = X_sub.std(axis=0, dtype=np.float64)
            std_val[std_val < 1e-8] = 1.0
            X[mask] = np.clip((X_sub - mean_val) / std_val, -10.0, 10.0)
    else:
        raise ValueError(
            f"Unsupported cross_standardization={standardization!r}; "
            "expected 'joint' or 'per_dataset'"
        )

    ref_types = set(np.unique(labels[is_ref]))
    tgt_types = set(np.unique(labels[is_tgt]))
    shared = ref_types & tgt_types  # known
    novel = tgt_types - ref_types  # target-private = novel
    ref_private = ref_types - tgt_types  # reference-only

    print("=== [CrossData] Open-set partition (by dataset membership) ===")
    print(f"  KNOWN (shared)      : {len(shared)}  {sorted(shared)}")
    print(f"  NOVEL (tgt-private) : {len(novel)}  {sorted(novel)}")
    print(f"  REF-private         : {len(ref_private)}  {sorted(ref_private)}")
    if len(novel) == 0:
        raise RuntimeError("[CrossData] 0 novel types; check ref/tgt direction.")

    indices_global, valid_ontology_mask = ontology_manager.map_labels(labels)
    indices_global = (
        indices_global.detach().cpu().long()
        if isinstance(indices_global, torch.Tensor)
        else torch.as_tensor(indices_global, dtype=torch.long)
    )
    valid_ontology_mask = (
        valid_ontology_mask.detach().cpu().bool()
        if isinstance(valid_ontology_mask, torch.Tensor)
        else torch.as_tensor(valid_ontology_mask, dtype=torch.bool)
    )

    novel_mask_global = np.isin(labels, list(novel))
    indices_global = indices_global.clone()
    indices_global[novel_mask_global] = -1

    valid_mask_np = valid_ontology_mask.cpu().numpy().astype(bool)
    src_valid_mask = valid_mask_np.copy()
    src_valid_mask[novel_mask_global] = False

    source_valid_classes = shared | ref_private
    src_select = is_ref & np.isin(labels, list(source_valid_classes)) & src_valid_mask
    src_idx = np.where(src_select)[0]

    if getattr(args, "source_label_ratio", 0.5) < 1.0:
        rng = np.random.RandomState(args.seed)
        src_class_labels = labels[src_idx]
        kept = []
        for cls in np.unique(src_class_labels):
            pos = src_idx[src_class_labels == cls]
            n_keep = max(
                getattr(args, "src_min_per_class", 10), int(len(pos) * args.source_label_ratio)
            )
            n_keep = min(n_keep, len(pos))
            kept.append(rng.choice(pos, n_keep, replace=False))
        src_idx = np.sort(np.concatenate(kept))
        print(f"[CrossData] Source subsampled: {len(src_idx)} cells")

    src_ds = TensorDataset(
        torch.tensor(X[src_idx], dtype=torch.float),
        torch.tensor(raw_X[src_idx], dtype=torch.float),
        torch.tensor(sf[src_idx], dtype=torch.float),
        indices_global[src_idx],
        torch.ones(len(src_idx), dtype=torch.bool),
    )

    unique_src_ids = torch.unique(indices_global[src_idx])
    unique_src_ids = unique_src_ids[unique_src_ids >= 0]

    target_valid_classes = shared | novel
    tgt_class_select = is_tgt & np.isin(labels, list(target_valid_classes))
    tgt_ontology_ok = valid_mask_np | novel_mask_global
    tgt_select = tgt_class_select & tgt_ontology_ok
    tgt_idx = np.where(tgt_select)[0]

    is_novel_gt = np.isin(labels[tgt_idx], list(novel)).astype(np.float32)
    tgt_eval_valid_mask = src_valid_mask[tgt_idx]

    novel_ratio = getattr(args, "novel_ratio", 1.0)
    if novel_ratio < 1.0 - 1e-6:
        rng_nr = np.random.RandomState(args.seed + 9999)
        novel_pos = np.where(is_novel_gt == 1)[0]
        known_pos = np.where(is_novel_gt == 0)[0]
        n_keep = max(1, int(len(novel_pos) * novel_ratio))
        kept_novel = rng_nr.choice(novel_pos, n_keep, replace=False)
        kept_all = np.sort(np.concatenate([known_pos, kept_novel]))
        tgt_idx = tgt_idx[kept_all]
        is_novel_gt = is_novel_gt[kept_all]
        tgt_eval_valid_mask = src_valid_mask[tgt_idx]
        print(f"[CrossData] Novel ratio={novel_ratio:.1f}: kept {n_keep}/{len(novel_pos)}")

    tgt_ds = TensorDataset(
        torch.tensor(X[tgt_idx], dtype=torch.float),
        torch.tensor(raw_X[tgt_idx], dtype=torch.float),
        torch.tensor(sf[tgt_idx], dtype=torch.float),
        indices_global[tgt_idx],
        torch.tensor(tgt_eval_valid_mask, dtype=torch.bool),
        torch.tensor(is_novel_gt, dtype=torch.float),
        torch.tensor(raw_label_ids[tgt_idx], dtype=torch.long),
    )

    print(
        f"[CrossData] Source: {len(src_ds)} cells | Target: {len(tgt_ds)} cells "
        f"(Novel GT: {int(is_novel_gt.sum())}, Known GT: {int((is_novel_gt==0).sum())})"
    )
    print(f"[CrossData] Unique src ontology IDs: {len(unique_src_ids)}")

    src_train_loader = DataLoader(
        src_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    src_eval_loader = DataLoader(
        src_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    tgt_train_ds = TensorDataset(*tgt_ds.tensors[:3])
    tgt_eval_loader = DataLoader(
        tgt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    tgt_train_loader = DataLoader(
        tgt_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    assert isinstance(tgt_train_loader.dataset, TensorDataset)
    assert len(tgt_train_loader.dataset.tensors) == 3, (
        "Cross-dataset target training dataset must contain exactly " "(x, raw_x, size_factor)."
    )
    assert len(tgt_eval_loader.dataset.tensors) == 7
    assert src_train_loader.drop_last is False
    assert tgt_train_loader.drop_last is False

    novel_list = sorted(list(novel))
    return (
        src_train_loader,
        src_eval_loader,
        tgt_eval_loader,
        tgt_train_loader,
        X.shape[1],
        unique_src_ids,
        novel_list,
        le,
    )
