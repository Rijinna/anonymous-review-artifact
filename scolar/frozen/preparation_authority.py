"""
train.py — scOLAR fixed-final protocol (Research-grade)
=========================================
论文逻辑主线：表示稳健性 → 原型几何 → 决策去偏 → 推理闭环 → 生物解释
对应 Loss   ：L_recon    → L_HCL    → L_dec-adv → MLS-alpha → LCC

修复清单（相对前一版本）：
  [F1] decision_adversarial_loss 废弃旧近似，改用 model 输出的 s_coarse_max
       （真实 coarse prototype 余弦相似度，物理梯度隔离，严格对齐论文 4.3）
  [F2] t_out['prototypes'][tvalid] -> t_out['prototypes']（维度 Bug，运行时 IndexError）
  [F3] LCC 矩阵乘法维度修正：fine_logits[:, src_ids] @ anc_mat[src_ids, :]
  [F4] model.set_ontology_info() 在模型创建后立即调用（buffer 注册）
  [F5] 训练结束后追加 validate_osr(epoch="FINAL")，触发完整 LCC 报告
  [F6] 第一次 reference-only calibrate 在预设 epoch 前执行，供该 epoch 的 boundary loss 使用
  [F7] HCL 动态 Margin Scheduling（论文创新点，早期大 m / 后期小 m 线性衰减）
  [F8] rec_tgt meter 正确记录（旧版从未 update）
  [F9] checkpoint 保存完整训练状态
  [F10] primary checkpoint 固定为最终 epoch；target GT 仅用于训练后 benchmark scoring
  [F11] 学习率采用预设 cosine annealing，仅由 epoch/总预算决定
  [F12] calibration held-out class draws 与训练 RNG 隔离，并持久化复用于全部 snapshots
  [F13] 2000-HVG 输入统一做 gene-wise z-score，并截断到 [-10, 10]
  [F14] Table S4 / primary protocol 由显式 profile 校验，避免静默参数漂移
  [F15] batch size 与 labelled ratio 默认对齐论文（1024 / 0.5），训练 loader 不丢尾批
  [F16] cosine LR 使用统一 100-epoch horizon；60-epoch cross runs 采用同一曲线前 60 步
  [F17] 双轨 evaluator：Track-1 GT-novel/L2/oracle-K KMeans + Track-2 predicted-novel/L2/Leiden
  [F18] Dev-B target-neighborhood consistency (TNC)：仅用未标注 target z、reference-derived
        ontology geometry 与 reference-calibrated alpha；不使用 target labels / K_novel，梯度只回到 z。
"""

import argparse
import hashlib
import json
import math
import os
import struct
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
from sklearn.metrics import (
    roc_auc_score, adjusted_rand_score, adjusted_mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans
from itertools import cycle
import scipy.sparse

from utils import OntologyManager
from models import scOLAR
from layers import scOLARLoss
from hierarchy_triplet_sampling import hierarchy_triplet_aug
from scipy.optimize import linear_sum_assignment


# =============================================================================
# 0. 工具类
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ONLY_HVG_PROTOCOL_VERSION = "source_only_hvg_partition_v1"
SOURCE_ONLY_HVG_ALLOWED_SEEDS = (1729, 2718)
SOURCE_ONLY_HVG_ALLOWED_FOLDS = (0, 1, 2)
SOURCE_ONLY_HVG_ALLOWED_HVG = (2000, 4000)
SOURCE_ONLY_HVG_DATASETS = (
    "Cao", "Quake_10x", "Quake_Smart-seq2", "Wagner", "Zeisel_2018"
)
SOURCE_ONLY_HVG_EXPECTED_MANIFEST_SHA256 = (
    "a6d390e18cd5f9b6b165c22c97575a374ed0dff95bf72d1782e945c484c91347"
)
SOURCE_ONLY_HVG_EXPECTED_PAYLOAD_SHA256 = (
    "106ccc0c077b6cb168636e9e7316501000778afdb241b967ec1e3e0aa020790e"
)
SOURCE_ONLY_HVG_EXPECTED_BASELINE_TRAIN_SHA256 = (
    "b941ea549ab5f8c85cec10dc5f50ab802beb74df440d49e022d473812978ffbd"
)
SOURCE_ONLY_HVG_EXPECTED_FREEZER_SHA256 = (
    "249fc83edbfd2481425ad739f7365b7de74283750fda7fa03999a6f324339ebf"
)
SOURCE_ONLY_HVG_EXPECTED_LEGACY_FOLDS_SHA256 = (
    "710a35950c9f68c943de956148139f5e58ad0aab8e2773f0af1b84e5dbd99eec"
)
SOURCE_ONLY_HVG_PREPROCESSING_VISIBILITY = (
    "production source universe -> frozen S/K/N; joint expression-only "
    "normalize/log/HVG/size-factor/gene-standardization on S∪K∪N; "
    "S labelled source and source-only calibration; K∪N unlabeled three-tensor target; "
    "real benchmark target and registry-outside cells structurally absent"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _encode_length_prefixed_utf8(value) -> bytes:
    payload = str(value).encode("utf-8")
    return struct.pack(">Q", len(payload)) + payload


def _ordered_string_list_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_encode_length_prefixed_utf8(value))
    return digest.hexdigest()


def _utf8_lexical(values) -> list:
    return sorted((str(value) for value in values), key=lambda item: item.encode("utf-8"))


def _read_h5ad_fail_closed(path: Path):
    """Read an h5ad with no mock/synthetic fallback under any profile."""
    absolute_path = Path(path).expanduser().resolve()
    try:
        return sc.read_h5ad(absolute_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read h5ad at '{absolute_path}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _finite(x):
    return x is None or torch.isfinite(x).all().item()

def cluster_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Implement Hungarian-matched clustering accuracy used by the benchmark."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def aligned_cluster_acc_and_macro(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int):
    """
    返回：
        acc_weighted : 匈牙利匹配后的全局加权准确率（等同于原 cluster_acc）
        acc_macro    : 匈牙利匹配后的 per-class macro 均值
        cm_aligned   : 对齐后的混淆矩阵 [n_clusters, n_clusters]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # 用匈牙利对齐结果重新映射 pred 标签
    pred_mapped = np.full_like(y_pred, fill_value=-1)
    label_map = dict(zip(row_ind, col_ind))
    for i, p in enumerate(y_pred):
        pred_mapped[i] = label_map.get(int(p), int(p))

    acc_weighted = w[row_ind, col_ind].sum() / y_pred.size

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, pred_mapped, labels=list(range(n_clusters)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    acc_macro = per_class.mean()

    return acc_weighted, acc_macro, cm    

def _l2_normalize_embeddings(z) -> np.ndarray:
    """Common external evaluator preprocessing: row-wise L2 normalization."""
    if isinstance(z, torch.Tensor):
        z_t = z.detach().cpu().float()
    else:
        z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    if z_t.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape={tuple(z_t.shape)}")
    return F.normalize(z_t, p=2, dim=1).numpy()

def common_oracle_kmeans(
    z,
    y_true_dense: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 10,
) -> dict:
    """
    Track 1: frozen common oracle-K evaluator.

    Only GT-novel cells are supplied by the caller. The evaluator performs:
      row-wise L2 normalization -> KMeans(K_novel) -> Hungarian scoring.
    K is oracle information used only for benchmark scoring. KMeans
    initialization is NOT selected with target labels: one deterministic
    sklearn KMeans fit is used with fixed random_state and n_init.
    """
    z_norm = _l2_normalize_embeddings(z)
    y_true_dense = np.asarray(y_true_dense, dtype=np.int64)
    if z_norm.shape[0] != y_true_dense.shape[0]:
        raise ValueError("Embedding/label length mismatch in common_oracle_kmeans")
    if n_clusters < 2 or z_norm.shape[0] < n_clusters:
        raise ValueError(
            f"Invalid oracle-K clustering problem: n_cells={z_norm.shape[0]}, K={n_clusters}"
        )

    km = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=int(random_state),
        n_init=int(n_init),
    )
    pred = km.fit_predict(z_norm)
    weighted, macro, cm = aligned_cluster_acc_and_macro(
        y_true_dense, pred, n_clusters
    )
    return {
        'pred': pred,
        'weighted_acc': float(weighted),
        'macro_acc': float(macro),
        'confusion': cm,
        'ari': float(adjusted_rand_score(y_true_dense, pred)),
        'ami': float(adjusted_mutual_info_score(y_true_dense, pred)),
        'nmi': float(normalized_mutual_info_score(y_true_dense, pred)),
        'inertia': float(km.inertia_),
        'k_novel': int(n_clusters),
        'kmeans_seed': int(random_state),
        'kmeans_n_init': int(n_init),
        'l2_normalized': True,
    }

def common_leiden(
    z,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 0,
) -> np.ndarray:
    """
    Track 2 common deployment clustering.

    No ground-truth K or labels are used to form clusters. Embeddings are
    L2-normalized first, then a fixed kNN graph + Leiden partition is applied.
    """
    z_norm = _l2_normalize_embeddings(z)
    n_cells = z_norm.shape[0]
    if n_cells < 2:
        return np.zeros(n_cells, dtype=np.int64)
    n_neighbors_eff = max(2, min(int(n_neighbors), n_cells - 1))
    adata_tmp = sc.AnnData(z_norm)
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=n_neighbors_eff,
        use_rep='X',
        metric='euclidean',
        random_state=int(random_state),
    )
    # Freeze Scanpy's changing defaults explicitly for reproducibility.
    sc.tl.leiden(
        adata_tmp,
        resolution=float(resolution),
        random_state=int(random_state),
        flavor='igraph',
        n_iterations=2,
        directed=False,
    )
    return adata_tmp.obs['leiden'].astype(int).values

def _safe_cluster_scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return {'ari': None, 'ami': None, 'nmi': None}
    return {
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'ami': float(adjusted_mutual_info_score(y_true, y_pred)),
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
    }

def compute_joint_overall(
    all_known_pred: list,
    all_known_true: list,
    novel_pred:     np.ndarray,
    nt_dense:       np.ndarray,
    n_known_classes: int,
) -> float:
    """
    复现 scBOL 的 cluster_acc(preds, targets) 语义：
    known 细胞使用分类器预测结果（索引 0 ~ n_known-1），
    novel 细胞使用 KMeans 预测结果偏移至 n_known ~ n_known+n_novel-1，
    合并后做一次全局匈牙利匹配，与 scBOL 的 overall_acc 定义对齐。
    """
    kp = torch.cat(all_known_pred).numpy()
    kt = torch.cat(all_known_true).numpy()

    np_shifted = novel_pred + n_known_classes
    nt_shifted = nt_dense   + n_known_classes

    all_pred = np.concatenate([kp, np_shifted])
    all_true = np.concatenate([kt, nt_shifted])

    return cluster_acc(all_pred, all_true)

class AverageMeter:
    """Track the sample-weighted running mean of one scalar metric."""

    def __init__(self, name: str, fmt: str = '.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# scBOL 对齐数据集的固定分割注册表
# 类型列表和顺序严格来自 bol_preprocess.py class_splitting_single()
# 切割比例来自 scBOL 论文 Table S4
# =============================================================================
SCBOL_CLASS_REGISTRY = {
    "Quake_10x": {
        "classes": [
            'B cell', 'T cell', 'alveolar macrophage', 'basal cell',
            'basal cell of epidermis', 'bladder cell', 'bladder urothelial cell',
            'blood cell', 'endothelial cell', 'epithelial cell', 'fibroblast',
            'granulocyte', 'granulocytopoietic cell', 'hematopoietic precursor cell',
            'hepatocyte', 'immature T cell', 'keratinocyte',
            'kidney capillary endothelial cell',
            'kidney collecting duct epithelial cell',
            'kidney loop of Henle ascending limb epithelial cell',
            'kidney proximal straight tubule epithelial cell',
            'late pro-B cell', 'leukocyte',
            'luminal epithelial cell of mammary gland', 'lung endothelial cell',
            'macrophage', 'mesenchymal cell', 'mesenchymal stem cell',
            'monocyte', 'natural killer cell', 'neuroendocrine cell',
            'non-classical monocyte', 'proerythroblast', 'promonocyte',
            'skeletal muscle satellite cell', 'stromal cell',
        ],
        "n_cs": 12, "n_cr": 12, "n_ct": 12,   # Table S4: |Cs|=12
    },
    "Quake_Smart-seq2": {
        "classes": [
            'B cell', 'Slamf1-negative multipotent progenitor cell', 'T cell',
            'astrocyte of the cerebral cortex', 'basal cell',
            'basal cell of epidermis', 'bladder cell', 'bladder urothelial cell',
            'blood cell', 'endothelial cell',
            'enterocyte of epithelium of large intestine', 'epidermal cell',
            'epithelial cell', 'epithelial cell of large intestine',
            'epithelial cell of proximal tubule', 'fibroblast', 'granulocyte',
            'hematopoietic precursor cell', 'hepatocyte', 'immature B cell',
            'immature T cell', 'keratinocyte', 'keratinocyte stem cell',
            'large intestine goblet cell', 'late pro-B cell', 'leukocyte',
            'luminal epithelial cell of mammary gland', 'lung endothelial cell',
            'macrophage', 'mesenchymal cell', 'mesenchymal stem cell',
            'mesenchymal stem cell of adipose', 'microglial cell', 'monocyte',
            'myeloid cell', 'naive B cell', 'neuron', 'oligodendrocyte',
            'oligodendrocyte precursor cell', 'pancreatic A cell',
            'pro-B cell', 'skeletal muscle satellite cell',
            'skeletal muscle satellite stem cell', 'stromal cell',
            'type B pancreatic cell',
        ],
        "n_cs": 15, "n_cr": 15, "n_ct": 15,   # Table S4: |Cs|=15
    },
    "Cao": {
        "classes": [
            'GABAergic neuron', 'cholinergic neuron',
            'ciliated olfactory receptor neuron', 'coelomocyte', 'epidermal cell',
            'germ line cell', 'glial cell', 'interneuron', 'muscle cell',
            'nasopharyngeal epithelial cell', 'neuron', 'seam cell',
            'sensory neuron', 'sheath cell', 'socket cell (sensu Nematoda)',
            'visceral muscle cell',
        ],
        "n_cs": 6, "n_cr": 4, "n_ct": 6,      # Table S4: |Cs|=6, |Cr|=4, |Ct|=6
    },
    "Wagner": {
        "classes": [
            'early embryonic cell', 'ectodermal cell', 'embryonic cell',
            'endodermal cell', 'epiblast cell', 'epidermal cell',
            'erythroid progenitor cell', 'lateral mesodermal cell',
            'mesodermal cell', 'midbrain dopaminergic neuron',
            'neural crest cell', 'neurecto-epithelial cell',
            'neuronal stem cell', 'spinal cord interneuron',
        ],
        "n_cs": 5, "n_cr": 4, "n_ct": 5,      # Table S4: |Cs|=5, |Cr|=4, |Ct|=5
    },
    "Zeisel_2018": {
        "classes": [
            'CNS neuron (sensu Vertebrata)', 'astrocyte', 'cerebellum neuron',
            'dentate gyrus of hippocampal formation granule cell',
            'endothelial cell of vascular tree', 'enteric neuron',
            'ependymal cell', 'glial cell', 'inhibitory interneuron',
            'microglial cell', 'neuroblast', 'oligodendrocyte',
            'peptidergic neuron', 'pericyte cell', 'peripheral sensory neuron',
            'perivascular macrophage', 'vascular associated smooth muscle cell',
        ],
        "n_cs": 6, "n_cr": 5, "n_ct": 6,      # Table S4: |Cs|=6, |Cr|=5, |Ct|=6
    },
}

# =============================================================================
# 1. 数据准备：Open-Set Protocol（论文 1.2 节）
# =============================================================================

def split_open_set_protocol(
    labels,
    dataset_name:   str   = None,   # ← 新增
    ratio_shared:   float = 0.6,
    ratio_src_priv: float = 0.2,
    seed:           int   = 42,
    n_cs_override:  int   = None,   # ← 新增
    n_cr_override:  int   = None,   # ← 新增
):
    # ── 固定分割（scBOL 对齐数据集）────────────────────────────
    if dataset_name in SCBOL_CLASS_REGISTRY:
        entry = SCBOL_CLASS_REGISTRY[dataset_name]
        existing = set(np.unique(labels))

        # 过滤数据中不存在的类型（健壮性保护）
        class_list = [c for c in entry["classes"] if c in existing]
        missing = set(entry["classes"]) - existing
        if missing:
            print(f"[Data] Warning: {len(missing)} classes in registry "
                  f"not found in data: {missing}")

        n_cs = n_cs_override if n_cs_override is not None else entry["n_cs"]
        n_cr = n_cr_override if n_cr_override is not None else entry["n_cr"]
        # n_ct 由剩余决定，不硬编码，防止过滤后数量变化

        shared_classes   = set(class_list[:n_cs])
        src_priv_classes = set(class_list[n_cs : n_cs + n_cr])
        novel_classes    = set(class_list[n_cs + n_cr:])

        print("=== [Open-Set Protocol] (fixed, scBOL-aligned) ===")

    # ── 比例分割（新数据集，seed 控制）──────────────────────────
    else:
        unique_types = np.unique(labels)
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_types)
        n          = len(unique_types)
        n_shared   = int(n * ratio_shared)
        n_src_priv = int(n * ratio_src_priv)
        shared_classes   = set(unique_types[:n_shared])
        src_priv_classes = set(unique_types[n_shared : n_shared + n_src_priv])
        novel_classes    = set(unique_types[n_shared + n_src_priv:])
        print("=== [Open-Set Protocol] (/-based, seed-controlled) ===")

    print(f"  Shared  (Cs): {len(shared_classes):>4} types")
    print(f"  Src Priv(Cr): {len(src_priv_classes):>4} types")
    print(f"  Novel   (Ct): {len(novel_classes):>4} types  <- OSR target")

    return shared_classes, src_priv_classes, novel_classes


def _source_only_manifest_error(message: str):
    raise RuntimeError(f"source_only_hvg manifest validation failed: {message}")


def _validate_source_only_hvg_manifest(args) -> dict:
    """Validate the frozen v1 manifest without regenerating any cell partition."""
    manifest_path = Path(args.source_only_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        _source_only_manifest_error(f"manifest does not exist: {manifest_path}")

    manifest_file_hash = _sha256_file(manifest_path)
    if manifest_file_hash != SOURCE_ONLY_HVG_EXPECTED_MANIFEST_SHA256:
        _source_only_manifest_error(
            "manifest file SHA256 mismatch: "
            f"expected={SOURCE_ONLY_HVG_EXPECTED_MANIFEST_SHA256}, "
            f"observed={manifest_file_hash}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _source_only_manifest_error(
            f"cannot parse {manifest_path}: {type(exc).__name__}: {exc}"
        )

    try:
        payload = manifest["scientific_payload"]
        declared_payload_hash = str(manifest["scientific_payload_sha256"])
        payload_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if declared_payload_hash != SOURCE_ONLY_HVG_EXPECTED_PAYLOAD_SHA256:
            _source_only_manifest_error(
                "declared scientific payload SHA256 differs from the frozen value"
            )
        if payload_hash != declared_payload_hash:
            _source_only_manifest_error(
                f"scientific payload hash mismatch: declared={declared_payload_hash}, "
                f"observed={payload_hash}"
            )
        if payload["protocol_version"] != SOURCE_ONLY_HVG_PROTOCOL_VERSION:
            _source_only_manifest_error(
                f"protocol_version={payload['protocol_version']!r}, "
                f"expected={SOURCE_ONLY_HVG_PROTOCOL_VERSION!r}"
            )
        if payload["protocol_status"] != "FROZEN_BEFORE_SOURCE_ONLY_HVG_TRAINING":
            _source_only_manifest_error("manifest protocol status is not frozen")

        provenance = payload["provenance"]
        if Path(provenance["project_root"]).resolve() != PROJECT_ROOT:
            _source_only_manifest_error("project_root differs from the authoritative Linux root")
        if provenance["train.py"]["sha256"] != SOURCE_ONLY_HVG_EXPECTED_BASELINE_TRAIN_SHA256:
            _source_only_manifest_error("pre-edit train.py provenance hash mismatch")

        freezer_info = provenance["freezer_script"]
        freezer_path = Path(freezer_info["path"]).resolve()
        if freezer_info["sha256"] != SOURCE_ONLY_HVG_EXPECTED_FREEZER_SHA256:
            _source_only_manifest_error("freezer provenance hash mismatch")
        if not freezer_path.is_file() or _sha256_file(freezer_path) != SOURCE_ONLY_HVG_EXPECTED_FREEZER_SHA256:
            _source_only_manifest_error(f"frozen freezer missing or modified: {freezer_path}")

        legacy_info = provenance["legacy_folds"]
        legacy_path = Path(legacy_info["path"]).resolve()
        if legacy_info["sha256"] != SOURCE_ONLY_HVG_EXPECTED_LEGACY_FOLDS_SHA256:
            _source_only_manifest_error("legacy folds provenance hash mismatch")
        if (not legacy_path.is_file()
                or _sha256_file(legacy_path) != SOURCE_ONLY_HVG_EXPECTED_LEGACY_FOLDS_SHA256):
            _source_only_manifest_error(f"legacy folds missing or modified: {legacy_path}")

        specification = payload["specification"]
        expected_key_components = [
            "protocol_version",
            "dataset",
            "development_seed",
            "fold_id",
            "canonical_class_id",
            "stable_cell_id",
        ]
        if specification["development_seeds"] != list(SOURCE_ONLY_HVG_ALLOWED_SEEDS):
            _source_only_manifest_error("development seed contract mismatch")
        if specification["datasets"] != list(SOURCE_ONLY_HVG_DATASETS):
            _source_only_manifest_error("dataset contract mismatch")
        if specification["fold_ids"] != list(SOURCE_ONLY_HVG_ALLOWED_FOLDS):
            _source_only_manifest_error("fold contract mismatch")
        if specification["screen_hvg_candidates"] != list(SOURCE_ONLY_HVG_ALLOWED_HVG):
            _source_only_manifest_error("HVG candidate contract mismatch")
        if specification["stable_cell_id"] != "str(adata.obs_names)":
            _source_only_manifest_error("stable cell identity contract mismatch")
        if specification["canonical_class_id"] != "cell_ontology_id":
            _source_only_manifest_error("canonical class identity contract mismatch")
        if not math.isclose(
            float(specification["pseudo_known_fraction"]), 0.20,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            _source_only_manifest_error("pseudo_known_fraction is not exactly 0.20")
        if specification["n_K_rule"] != "n_class // 5":
            _source_only_manifest_error("n_K rounding rule mismatch")
        if specification["partition_key_components"] != expected_key_components:
            _source_only_manifest_error("partition key components differ from v1")
        if specification["hvg_in_partition_key"] is not False:
            _source_only_manifest_error("hvg must be absent from the partition key")

        preprocessing_contract = payload["later_preprocessing_sanity_contract"]
        required_preprocessing_contract = (
            "fixed_dataset_seed_hvg_selected_HVG_ordered_gene_list_hash_identical_across_folds",
            "fixed_dataset_seed_fold_partition_hash_identical_for_HVG2000_and_HVG4000",
            "changing_hvg_must_not_regenerate_partition",
        )
        for field in required_preprocessing_contract:
            if preprocessing_contract[field] is not True:
                _source_only_manifest_error(
                    f"later preprocessing sanity contract is not true: {field}"
                )

        global_validation = payload["global_validation"]
        required_global_true = (
            "all_real_benchmark_target_intersections_zero",
            "all_row_order_permutation_tests_passed",
            "all_HVG2000_HVG4000_partition_hash_pairs_identical",
            "hvg_absent_from_partition_key_components",
            "legacy_held_out_class_lists_copied_exactly",
        )
        if int(global_validation["partition_count"]) != 30:
            _source_only_manifest_error("manifest does not contain exactly 30 partitions")
        for field in required_global_true:
            if global_validation[field] is not True:
                _source_only_manifest_error(f"global invariant is not true: {field}")

        dataset_payload = payload["datasets"][args.dataset]
        seed_payload = dataset_payload["development_seeds"][str(int(args.seed))]
        fold_payload = seed_payload["folds"][str(int(args.source_only_fold))]
    except (KeyError, TypeError, ValueError) as exc:
        _source_only_manifest_error(
            f"missing or malformed required field: {type(exc).__name__}: {exc}"
        )

    source_universe = seed_payload["source_universe"]
    source_ids = [str(value) for value in source_universe["ordered_stable_ids"]]
    if len(source_ids) != int(source_universe["count"]):
        _source_only_manifest_error("source-universe count/list length mismatch")
    if len(source_ids) != len(set(source_ids)):
        _source_only_manifest_error("source-universe stable IDs are not unique")
    if _ordered_string_list_sha256(source_ids) != source_universe["ordered_list_sha256"]:
        _source_only_manifest_error("source-universe production-order hash mismatch")
    source_set_hash = _ordered_string_list_sha256(_utf8_lexical(source_ids))
    if source_set_hash != source_universe["set_sha256"]:
        _source_only_manifest_error("source-universe set hash mismatch")
    source_set = set(source_ids)

    def validate_role(role_name: str, role_payload: dict) -> tuple:
        role_ids = [str(value) for value in role_payload["ordered_stable_ids"]]
        if len(role_ids) != int(role_payload["count"]):
            _source_only_manifest_error(f"{role_name} count/list length mismatch")
        if len(role_ids) != len(set(role_ids)):
            _source_only_manifest_error(f"{role_name} stable IDs are not unique")
        if role_ids != _utf8_lexical(role_ids):
            _source_only_manifest_error(f"{role_name} IDs are not in frozen UTF-8 order")
        observed_hash = _ordered_string_list_sha256(role_ids)
        if observed_hash != role_payload["ordered_list_sha256"]:
            _source_only_manifest_error(f"{role_name} ordered-list hash mismatch")
        return role_ids, observed_hash

    role_ids = {}
    role_hashes = {}
    for role in ("S", "K", "N"):
        role_ids[role], role_hashes[role] = validate_role(
            role, fold_payload["roles"][role]
        )
    role_sets = {role: set(ids) for role, ids in role_ids.items()}
    if role_sets["S"] & role_sets["K"] or role_sets["S"] & role_sets["N"] \
            or role_sets["K"] & role_sets["N"]:
        _source_only_manifest_error("S/K/N are not pairwise disjoint")
    if set().union(*role_sets.values()) != source_set:
        _source_only_manifest_error("S∪K∪N does not equal the frozen source universe")

    class_registry = dataset_payload["class_registry"]
    source_class_names = [str(value) for value in class_registry["source_class_names"]]
    canonical_class_ids = {
        str(name): str(class_id)
        for name, class_id in class_registry["canonical_class_ids"].items()
    }
    held_out_names = [str(value) for value in fold_payload["held_out_class_names"]]
    held_out_ids = [str(value) for value in fold_payload["held_out_canonical_class_ids"]]
    legacy_held_out = dataset_payload["legacy_held_out_class_names_by_fold"][
        str(int(args.source_only_fold))
    ]
    if held_out_names != legacy_held_out:
        _source_only_manifest_error("held-out class names differ from the frozen legacy fold")
    if held_out_ids != [canonical_class_ids[name] for name in held_out_names]:
        _source_only_manifest_error("held-out canonical class IDs are inconsistent")
    expected_retained = [name for name in source_class_names if name not in set(held_out_names)]
    if fold_payload["retained_class_names"] != expected_retained:
        _source_only_manifest_error("retained class list is inconsistent")

    invariants = fold_payload["invariant_validation"]
    required_partition_true = (
        "S_K_N_pairwise_disjoint",
        "S_union_K_union_N_equals_source_universe",
        "real_benchmark_target_intersection_zero",
        "registry_outside_cells_absent",
        "N_class_set_equals_legacy_fold",
        "K_S_class_and_canonical_id_preserved",
        "row_order_permutation_partition_and_role_hashes_identical",
        "HVG2000_HVG4000_partition_hashes_identical",
    )
    if int(invariants["real_benchmark_target_intersection_count"]) != 0:
        _source_only_manifest_error("frozen target-intersection count is nonzero")
    for field in required_partition_true:
        if invariants[field] is not True:
            _source_only_manifest_error(f"partition invariant is not true: {field}")
    if invariants["hvg_in_partition_key"] is not False:
        _source_only_manifest_error("partition invariant says hvg entered the key")
    if invariants["partition_function_accepts_hvg_argument"] is not False:
        _source_only_manifest_error("partition function must not accept hvg")

    partition_core = {
        "protocol_version": SOURCE_ONLY_HVG_PROTOCOL_VERSION,
        "dataset": args.dataset,
        "development_seed": int(args.seed),
        "fold_id": int(args.source_only_fold),
        "source_universe_set_sha256": source_universe["set_sha256"],
        "held_out_canonical_class_ids": held_out_ids,
        "roles": {
            role: {
                "count": int(fold_payload["roles"][role]["count"]),
                "ordered_list_sha256": role_hashes[role],
            }
            for role in ("S", "K", "N")
        },
    }
    observed_partition_hash = hashlib.sha256(_canonical_json_bytes(partition_core)).hexdigest()
    if observed_partition_hash != fold_payload["partition_hash"]:
        _source_only_manifest_error("partition hash does not match its canonical core")
    candidate_hashes = fold_payload["candidate_partition_hashes"]
    if set(candidate_hashes) != {"2000", "4000"}:
        _source_only_manifest_error("candidate partition-hash keys are not exactly 2000/4000")
    if any(value != observed_partition_hash for value in candidate_hashes.values()):
        _source_only_manifest_error("HVG candidates do not share one partition hash")

    # The source universe is stored once per dataset/seed, outside the folds. Verify
    # every fold consumes exactly this same universe, which makes preprocessing/HVG
    # fitting structurally fold-independent for a fixed dataset+seed+hvg.
    for other_fold in SOURCE_ONLY_HVG_ALLOWED_FOLDS:
        other = seed_payload["folds"][str(other_fold)]
        other_union = set()
        for role in ("S", "K", "N"):
            ids, _ = validate_role(f"fold{other_fold}.{role}", other["roles"][role])
            other_union.update(ids)
        if other_union != source_set:
            _source_only_manifest_error(
                f"fold {other_fold} does not consume the common source universe"
            )

    return {
        "manifest_path": manifest_path,
        "manifest_file_sha256": manifest_file_hash,
        "scientific_payload_sha256": payload_hash,
        "dataset_payload": dataset_payload,
        "seed_payload": seed_payload,
        "fold_payload": fold_payload,
        "source_ids": source_ids,
        "source_set_sha256": source_universe["set_sha256"],
        "source_ordered_list_sha256": source_universe["ordered_list_sha256"],
        "role_ids": role_ids,
        "role_hashes": role_hashes,
        "source_class_names": source_class_names,
        "canonical_class_ids": canonical_class_ids,
        "held_out_class_names": held_out_names,
        "held_out_canonical_class_ids": held_out_ids,
        "partition_hash": observed_partition_hash,
    }


def prepare_source_only_hvg_episode(args, ontology_manager: OntologyManager):
    """Read, verify and preprocess one frozen source-only development episode."""
    context = _validate_source_only_hvg_manifest(args)
    dataset_payload = context["dataset_payload"]
    h5ad_info = dataset_payload["h5ad"]
    h5ad_path = Path(h5ad_info["path"]).resolve()
    if not h5ad_path.is_file():
        raise RuntimeError(f"Authoritative h5ad does not exist: {h5ad_path}")
    observed_h5ad_hash = _sha256_file(h5ad_path)
    if observed_h5ad_hash != h5ad_info["sha256"]:
        raise RuntimeError(
            f"Authoritative h5ad SHA256 mismatch for {args.dataset}: "
            f"expected={h5ad_info['sha256']}, observed={observed_h5ad_hash}"
        )
    adata = _read_h5ad_fail_closed(h5ad_path)
    if int(adata.n_obs) != int(h5ad_info["n_obs"]):
        raise RuntimeError("Authoritative h5ad n_obs differs from the frozen manifest")
    if "cell_ontology_class" not in adata.obs or "cell_ontology_id" not in adata.obs:
        raise RuntimeError(
            "source_only_hvg requires cell_ontology_class and cell_ontology_id in adata.obs"
        )
    if adata.obs.index.hasnans:
        raise RuntimeError("source_only_hvg requires non-missing adata.obs_names")
    stable_ids = [str(value) for value in adata.obs_names]
    if any(not value for value in stable_ids):
        raise RuntimeError("source_only_hvg encountered an empty stable obs_name")
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("source_only_hvg requires unique str(adata.obs_names)")

    id_to_row = {stable_id: index for index, stable_id in enumerate(stable_ids)}
    missing_source_ids = [
        stable_id for stable_id in context["source_ids"] if stable_id not in id_to_row
    ]
    if missing_source_ids:
        raise RuntimeError(
            f"Frozen source IDs missing from h5ad: n={len(missing_source_ids)}, "
            f"first={missing_source_ids[:3]}"
        )
    source_rows = np.asarray(
        [id_to_row[stable_id] for stable_id in context["source_ids"]], dtype=np.int64
    )
    actual_episode_ids = [stable_ids[index] for index in source_rows]
    if actual_episode_ids != context["source_ids"]:
        raise RuntimeError("Stable-ID episode reconstruction changed frozen source order")

    labels_all = adata.obs["cell_ontology_class"]
    ontology_ids_all = adata.obs["cell_ontology_id"]
    if labels_all.iloc[source_rows].isna().any():
        raise RuntimeError("Frozen source universe contains a missing cell_ontology_class")
    if ontology_ids_all.iloc[source_rows].isna().any():
        raise RuntimeError("Frozen source universe contains a missing cell_ontology_id")
    labels_source = np.asarray(
        [str(value) for value in labels_all.iloc[source_rows]], dtype=object
    )
    ontology_ids_source = np.asarray(
        [str(value) for value in ontology_ids_all.iloc[source_rows]], dtype=object
    )
    source_id_to_label = dict(zip(context["source_ids"], labels_source))
    source_id_to_ontology = dict(zip(context["source_ids"], ontology_ids_source))
    source_class_set = set(context["source_class_names"])
    unexpected_classes = set(labels_source.tolist()) - source_class_set
    if unexpected_classes:
        raise RuntimeError(
            f"Registry-outside classes entered source_only_hvg: {sorted(unexpected_classes)}"
        )
    for stable_id in context["source_ids"]:
        class_name = source_id_to_label[stable_id]
        expected_class_id = context["canonical_class_ids"][class_name]
        if source_id_to_ontology[stable_id] != expected_class_id:
            raise RuntimeError(
                f"Canonical class ID mismatch for cell {stable_id}: "
                f"expected={expected_class_id}, observed={source_id_to_ontology[stable_id]}"
            )

    role_sets = {role: set(ids) for role, ids in context["role_ids"].items()}
    source_set = set(context["source_ids"])
    if set().union(*role_sets.values()) != source_set:
        raise RuntimeError("Runtime S∪K∪N differs from the frozen source universe")
    if role_sets["S"] & role_sets["K"] or role_sets["S"] & role_sets["N"] \
            or role_sets["K"] & role_sets["N"]:
        raise RuntimeError("Runtime S/K/N are not pairwise disjoint")
    held_out_set = set(context["held_out_class_names"])
    retained_set = source_class_set - held_out_set
    if {source_id_to_label[value] for value in context["role_ids"]["N"]} != held_out_set:
        raise RuntimeError("Runtime N class set differs from the frozen held-out fold")
    for role in ("S", "K"):
        if not {source_id_to_label[value] for value in context["role_ids"][role]} <= retained_set:
            raise RuntimeError(f"Runtime {role} contains a held-out or registry-outside class")
    expected_n_ids = {
        stable_id for stable_id in context["source_ids"]
        if source_id_to_label[stable_id] in held_out_set
    }
    if role_sets["N"] != expected_n_ids:
        raise RuntimeError("Runtime N does not contain all and only held-out source cells")

    per_retained = context["fold_payload"]["per_retained_class"]
    for class_name in context["fold_payload"]["retained_class_names"]:
        class_payload = per_retained[class_name]
        class_k = _utf8_lexical(
            value for value in context["role_ids"]["K"]
            if source_id_to_label[value] == class_name
        )
        class_s = _utf8_lexical(
            value for value in context["role_ids"]["S"]
            if source_id_to_label[value] == class_name
        )
        n_class = len(class_k) + len(class_s)
        if n_class != int(class_payload["n_class"]):
            raise RuntimeError(f"Runtime retained-class count mismatch: {class_name}")
        if len(class_k) != n_class // 5 or len(class_k) != int(class_payload["n_K"]):
            raise RuntimeError(f"Runtime n_K rule mismatch: {class_name}")
        if len(class_s) != int(class_payload["n_S"]):
            raise RuntimeError(f"Runtime n_S mismatch: {class_name}")
        if _ordered_string_list_sha256(class_k) != class_payload["K_ordered_list_sha256"]:
            raise RuntimeError(f"Runtime per-class K hash mismatch: {class_name}")
        if _ordered_string_list_sha256(class_s) != class_payload["S_ordered_list_sha256"]:
            raise RuntimeError(f"Runtime per-class S hash mismatch: {class_name}")

    # Subset before every fitted preprocessing operation. The source universe is
    # shared across folds, so neither held-out identities nor fold-specific roles
    # can affect normalize/log/HVG/size-factor/z-score statistics.
    episode = adata[source_rows].copy()
    if [str(value) for value in episode.obs_names] != context["source_ids"]:
        raise RuntimeError("Episode subsetting did not preserve frozen source-universe order")
    if "counts" not in episode.layers:
        episode.layers["counts"] = episode.X.copy()
    if int(episode.n_vars) < int(args.hvg):
        raise RuntimeError(
            f"Requested hvg={args.hvg} exceeds n_vars={episode.n_vars} in {h5ad_path}"
        )

    sc.pp.normalize_total(episode, target_sum=1e4)
    sc.pp.log1p(episode)
    sc.pp.highly_variable_genes(episode, n_top_genes=args.hvg, subset=False)
    hv_mask = np.asarray(episode.var["highly_variable"].values, dtype=bool)
    if int(hv_mask.sum()) != int(args.hvg):
        raise RuntimeError(
            f"HVG selection returned {int(hv_mask.sum())} genes, expected {args.hvg}"
        )
    if episode.var.index.hasnans or not episode.var_names.is_unique:
        raise RuntimeError("Selected-HVG hashing requires unique, non-missing adata.var_names")
    selected_hvg_names = [str(value) for value in episode.var_names[hv_mask]]
    if any(not value for value in selected_hvg_names):
        raise RuntimeError("Selected-HVG list contains an empty gene identifier")
    if len(selected_hvg_names) != len(set(selected_hvg_names)):
        raise RuntimeError("Selected-HVG identifiers are not unique after string conversion")
    selected_hvg_hash = _ordered_string_list_sha256(selected_hvg_names)

    X_full = episode.X.toarray() if scipy.sparse.issparse(episode.X) else np.asarray(episode.X)
    raw_full = episode.layers["counts"]
    raw_full = raw_full.toarray() if scipy.sparse.issparse(raw_full) else np.asarray(raw_full)
    sf_full = raw_full.sum(axis=1, keepdims=True).astype(np.float32)
    sf_full = np.maximum(sf_full, 1.0)
    sf = (sf_full / (np.median(sf_full) + 1e-8)).astype(np.float32)
    sf = np.clip(sf, 0.1, 10.0)
    X = X_full[:, hv_mask].astype(np.float32)
    feature_mean = X.mean(axis=0, dtype=np.float64)
    feature_std = X.std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-8] = 1.0
    X = ((X - feature_mean) / feature_std).astype(np.float32)
    X = np.clip(X, -10.0, 10.0)
    raw_X = raw_full[:, hv_mask].astype(np.float32)

    episode_id_to_row = {
        stable_id: index for index, stable_id in enumerate(context["source_ids"])
    }
    role_rows = {
        role: np.asarray(
            [episode_id_to_row[value] for value in context["role_ids"][role]],
            dtype=np.int64,
        )
        for role in ("S", "K", "N")
    }

    cl_to_global_index = {
        str(class_id): index for index, class_id in enumerate(ontology_manager.all_terms)
    }
    missing_cl_ids = set(context["canonical_class_ids"].values()) - set(cl_to_global_index)
    if missing_cl_ids:
        raise RuntimeError(
            f"Compiled ontology lacks source canonical IDs: {sorted(missing_cl_ids)}"
        )
    class_to_global = {
        class_name: cl_to_global_index[class_id]
        for class_name, class_id in context["canonical_class_ids"].items()
    }
    global_label_ids = np.asarray(
        [class_to_global[str(value)] for value in labels_source], dtype=np.int64
    )

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(np.asarray(context["source_class_names"], dtype=object))
    raw_label_ids = le.transform(labels_source).astype(np.int64)

    s_rows = role_rows["S"]
    source_ds = TensorDataset(
        torch.tensor(X[s_rows], dtype=torch.float),
        torch.tensor(raw_X[s_rows], dtype=torch.float),
        torch.tensor(sf[s_rows], dtype=torch.float),
        torch.tensor(global_label_ids[s_rows], dtype=torch.long),
        torch.ones(len(s_rows), dtype=torch.bool),
    )
    src_train_loader = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=0, pin_memory=False,
    )
    src_eval_loader = DataLoader(
        source_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=False,
    )

    # Evaluation-only target ordering is frozen K followed by N. Only the first
    # three tensors are shared with the training path; GT fields remain solely in
    # the separate post-final evaluation dataset.
    target_rows = np.concatenate([role_rows["K"], role_rows["N"]])
    n_k = len(role_rows["K"])
    n_n = len(role_rows["N"])
    target_global_ids = global_label_ids[target_rows].copy()
    target_global_ids[n_k:] = -1
    target_eval_ds = TensorDataset(
        torch.tensor(X[target_rows], dtype=torch.float),
        torch.tensor(raw_X[target_rows], dtype=torch.float),
        torch.tensor(sf[target_rows], dtype=torch.float),
        torch.tensor(target_global_ids, dtype=torch.long),
        torch.tensor([True] * n_k + [False] * n_n, dtype=torch.bool),
        torch.tensor([0.0] * n_k + [1.0] * n_n, dtype=torch.float),
        torch.tensor(raw_label_ids[target_rows], dtype=torch.long),
    )
    target_train_ds = TensorDataset(*target_eval_ds.tensors[:3])
    tgt_train_loader = DataLoader(
        target_train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=0, pin_memory=False,
    )
    post_final_eval_loader = DataLoader(
        target_eval_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=False,
    )
    if len(source_ds.tensors) != 5:
        raise RuntimeError("source_only_hvg source loader must contain exactly five tensors")
    if len(target_train_ds.tensors) != 3:
        raise RuntimeError(
            "source_only_hvg target training loader must contain exactly "
            "(x, raw_x, size_factor)"
        )
    if len(target_eval_ds.tensors) != 7:
        raise RuntimeError("source_only_hvg post-final evaluation dataset schema mismatch")

    unique_src_ids = torch.unique(torch.tensor(global_label_ids[s_rows], dtype=torch.long))
    unique_src_ids = unique_src_ids[unique_src_ids >= 0]
    role_provenance = {
        role: {
            "count": int(context["fold_payload"]["roles"][role]["count"]),
            "ordered_list_sha256": context["role_hashes"][role],
        }
        for role in ("S", "K", "N")
    }
    episode_provenance = {
        "profile": "source_only_hvg",
        "partition_protocol_version": SOURCE_ONLY_HVG_PROTOCOL_VERSION,
        "partition_manifest_path": str(context["manifest_path"]),
        "partition_manifest_file_sha256": context["manifest_file_sha256"],
        "canonical_scientific_payload_sha256": context["scientific_payload_sha256"],
        "partition_manifest_baseline_train_sha256": SOURCE_ONLY_HVG_EXPECTED_BASELINE_TRAIN_SHA256,
        "runtime_train_py_sha256": _sha256_file(Path(__file__).resolve()),
        "dataset": args.dataset,
        "development_seed": int(args.seed),
        "fold": int(args.source_only_fold),
        "requested_hvg": int(args.hvg),
        "authoritative_h5ad_path": str(h5ad_path),
        "authoritative_h5ad_sha256": observed_h5ad_hash,
        "source_universe_count": len(context["source_ids"]),
        "source_universe_ordered_list_sha256": context["source_ordered_list_sha256"],
        "source_universe_set_sha256": context["source_set_sha256"],
        "roles": role_provenance,
        "held_out_class_names": context["held_out_class_names"],
        "held_out_canonical_class_ids": context["held_out_canonical_class_ids"],
        "partition_hash": context["partition_hash"],
        "selected_hvg_count": len(selected_hvg_names),
        "selected_hvg_ordered_gene_list_sha256": selected_hvg_hash,
        "selected_hvg_hash_encoding": "SHA256(uint64_big_endian_len || UTF8(gene_id), repeated)",
        "selected_hvg_scope_source_universe_set_sha256": context["source_set_sha256"],
        "fixed_dataset_seed_hvg_gene_hash_identical_across_folds_by_construction": True,
        "hvg_absent_from_partition_key": True,
        "preprocessing_visibility": SOURCE_ONLY_HVG_PREPROCESSING_VISIBILITY,
        "source_loader_tensor_schema": ["x", "raw_x", "size_factor", "ontology_id", "valid_mask"],
        "target_training_tensor_schema": ["x", "raw_x", "size_factor"],
        "target_training_labels_present": False,
        "benchmark_target_cells_present": False,
        "registry_outside_cells_present": False,
        "calibration_uses": "S only",
        "checkpoint_selection": f"fixed_final_epoch_{args.epochs}",
        "target_gt_used_for_selection": False,
        "post_final_gt_roles": {"known": "K", "novel": "N"},
        "runtime_invariants": {
            "S_K_N_pairwise_disjoint": True,
            "S_union_K_union_N_equals_source_universe": True,
            "all_expected_ids_exist_exactly_once": True,
            "no_unexpected_episode_ids": True,
            "three_tensor_target_training_contract": True,
            "real_benchmark_target_intersection_zero_from_frozen_manifest": True,
        },
    }
    print(
        "[source_only_hvg] Frozen episode validated: "
        f"dataset={args.dataset} seed={args.seed} fold={args.source_only_fold} "
        f"hvg={args.hvg} S/K/N={len(s_rows)}/{n_k}/{n_n} "
        f"partition={context['partition_hash']} selected_hvg={selected_hvg_hash}"
    )
    return (
        src_train_loader,
        src_eval_loader,
        post_final_eval_loader,
        tgt_train_loader,
        X.shape[1],
        unique_src_ids,
        context["held_out_class_names"],
        le,
        episode_provenance,
    )


def prepare_datasets_robust(args, ontology_manager: OntologyManager):
    """
    构建带有严谨 Ground-Truth 标记的单细胞 OSR DataLoader。

    返回 8 元组：
        src_train_loader : Source training loader（5 元素，shuffle=True）
        src_eval_loader  : Source calibration loader（5 元素，完整、确定性）
        tgt_eval_loader  : Target benchmark loader（7 元素，含 GT；仅训练后评估）
        tgt_train_loader : Target unsupervised loader（仅 x/raw/sf，不含任何 GT）
        input_dim        : HVG 基因数（模型输入维度）
        unique_src_ids   : [n_src] long tensor，训练集实际出现的本体 node ID
        novel_list       : 固定顺序的 target-private 类别名
        le               : 仅供训练后 benchmark scoring 使用的 LabelEncoder

    编码器输入预处理与 Supplementary Table S1/S1.2 对齐：
    normalize_total → log1p → 2000 HVGs → 按基因在 Dr∪Dt 上 z-score → clip[-10, 10]。
    raw counts 始终单独保留，作为 ZINB likelihood 的重构目标。
    """
    print(f"\n=== [Data] Loading: {args.dataset} ===")

    # 1. 读取数据：所有 manuscript/production profile 均 fail closed，禁止 mock fallback。
    adata = _read_h5ad_fail_closed(PROJECT_ROOT / "data" / f"{args.dataset}.h5ad")

    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()

    # 2. 预处理：normalize -> log1p -> HVG 标记（subset=False），手动切片
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.hvg, subset=False)
    hv_mask = adata.var['highly_variable'].values                      # [n_genes] bool

    X_full = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)

    raw_full = adata.layers['counts']
    raw_full = raw_full.toarray() if scipy.sparse.issparse(raw_full) else np.array(raw_full)

    # 先用全基因 counts 算 size factor
    sf_full = raw_full.sum(axis=1, keepdims=True).astype(np.float32)
    sf_full = np.maximum(sf_full, 1.0)  # 防止极端情况下出现 0
    sf = (sf_full / (np.median(sf_full) + 1e-8)).astype(np.float32)
    sf = np.clip(sf, 0.1, 10.0)  # 防止极端值

    # 再切 HVG，并按论文协议在 reference+target 并集上做 gene-wise z-score。
    # 这是 transductive preprocessing：只使用无标签表达值，不使用任何 target label。
    X = X_full[:, hv_mask].astype(np.float32)
    feature_mean = X.mean(axis=0, dtype=np.float64)
    feature_std = X.std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-8] = 1.0
    X = ((X - feature_mean) / feature_std).astype(np.float32)
    X = np.clip(X, -10.0, 10.0)
    raw_X = raw_full[:, hv_mask].astype(np.float32)

    print("[Data] Encoder input standardized gene-wise on Dr∪Dt and clipped to [-10, 10].")
    labels = adata.obs['cell_ontology_class'].values

    # ── 新增：保存原始类别编码，供 novel_acc 评估用 ──
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    raw_label_ids = le.fit_transform(labels)

    # 3. Open-Set 协议划分
    shared, src_priv, novel = split_open_set_protocol(
        labels,
        dataset_name   = args.dataset,     # ← 加这一行
        ratio_shared   = args.ratio_shared,
        ratio_src_priv = args.ratio_src_priv,
        seed           = args.seed,
        n_cs_override  = getattr(args, 'n_cs_override', None),
        n_cr_override  = getattr(args, 'n_cr_override', None),
    )
    source_valid_classes = shared | src_priv
    target_valid_classes = shared | novel

    # 4. Ontology 映射
    indices_global, valid_ontology_mask = ontology_manager.map_labels(labels)

    # 强制转成 CPU tensor，避免 DataLoader worker 触发 CUDA 初始化
    if isinstance(indices_global, torch.Tensor):
        indices_global = indices_global.detach().cpu()
    else:
        indices_global = torch.as_tensor(indices_global, dtype=torch.long)

    if isinstance(valid_ontology_mask, torch.Tensor):
        valid_ontology_mask = valid_ontology_mask.detach().cpu()
    else:
        valid_ontology_mask = torch.as_tensor(valid_ontology_mask, dtype=torch.bool)

    indices_global = indices_global.long()
    valid_ontology_mask = valid_ontology_mask.bool()

    # Dataset-label aliases that are semantically identical to canonical CL names
    # but are not recovered by exact-name/exact-synonym matching in the compiled CL.
    # Keep these explicit and dataset-scoped for reproducibility.
    CURATED_CL_NAME_ALIASES = {
        "Wagner": {
            "early embryonic cell": "early embryonic cell (metazoa)",  # CL:0000007
            "embryonic cell": "embryonic cell (metazoa)",              # CL:0002321
        },
    }
    curated_aliases = CURATED_CL_NAME_ALIASES.get(args.dataset, {})
    if curated_aliases:
        cl_name_to_index = {str(name): int(idx) for idx, name in ontology_manager.id2name.items()}
        for dataset_label, canonical_cl_name in curated_aliases.items():
            cell_mask = np.asarray(labels == dataset_label)
            if not cell_mask.any():
                continue
            if canonical_cl_name not in cl_name_to_index:
                raise RuntimeError(
                    f"Curated ontology alias target missing from compiled CL: "
                    f"{dataset_label!r} -> {canonical_cl_name!r}. "
                    "Check data/cl.pt release before running."
                )
            global_idx = cl_name_to_index[canonical_cl_name]
            indices_global[cell_mask] = global_idx
            valid_ontology_mask[cell_mask] = True
            print(
                f"[Ontology mapping] curated alias: {dataset_label!r} -> "
                f"{canonical_cl_name!r} (global_idx={global_idx}, n={int(cell_mask.sum())})"
            )

    # Novel 标签强制脱钩：即使名称存在于本体，也将其 ID 置 -1，
    # 防止 OSR 退化为 Closed-Set。
    novel_mask_global = np.isin(labels, list(novel))
    indices_global    = indices_global.clone()
    indices_global[novel_mask_global] = -1

    valid_mask_np  = valid_ontology_mask.cpu().numpy().astype(bool)
    src_valid_mask = valid_mask_np.copy()
    src_valid_mask[novel_mask_global] = False       # novel 永不参与监督

    # 5. Benchmark-exclusive source/target cell allocation.
    # Match the released scBOL intra-dataset protocol exactly:
    #   shared cell      -> source with probability source_label_ratio, otherwise target
    #   source-private   -> source (100%)
    #   novel            -> target (100%)
    # This makes source and target disjoint and avoids discarding source-private cells.
    rng_split = np.random.RandomState(args.seed)
    src_idx_list = []
    tgt_idx_list = []
    for i, cls in enumerate(labels):
        if cls in shared:
            if rng_split.rand() < float(args.source_label_ratio):
                src_idx_list.append(i)
            else:
                tgt_idx_list.append(i)
        elif cls in src_priv:
            src_idx_list.append(i)
        elif cls in novel:
            tgt_idx_list.append(i)

    src_idx = np.asarray(src_idx_list, dtype=np.int64)
    tgt_idx = np.asarray(tgt_idx_list, dtype=np.int64)

    # scOLAR requires ontology IDs for every supervised source cell.  Do not
    # silently drop cells here, because doing so would break benchmark alignment.
    bad_src = src_idx[~src_valid_mask[src_idx]]
    if bad_src.size:
        bad_names = np.unique(labels[bad_src]).tolist()
        raise RuntimeError(
            "Benchmark source split contains cells without usable ontology IDs: "
            f"n={bad_src.size}, classes={bad_names}. "
            "Resolve ontology mapping rather than silently changing the split."
        )

    if np.intersect1d(src_idx, tgt_idx).size != 0:
        raise RuntimeError("Source/target overlap detected after benchmark-exclusive split")

    print(f"[Data] Benchmark-exclusive split: source={len(src_idx)} target={len(tgt_idx)} "
          f"(shared->source p={args.source_label_ratio:.2f}; src-private kept 100%)")

    src_ds = TensorDataset(
        torch.tensor(X[src_idx],     dtype=torch.float),
        torch.tensor(raw_X[src_idx], dtype=torch.float),
        torch.tensor(sf[src_idx],    dtype=torch.float),
        indices_global[src_idx],                                       # 本体 ID（long）
        torch.ones(len(src_idx), dtype=torch.bool),                    # valid_mask（全 True）
    )

    # unique_src_ids：训练集实际使用的本体 node ID
    # 用于 model buffer 中 coarse prototype 聚合 & LCC 矩阵切片
    unique_src_ids = torch.unique(indices_global[src_idx])
    unique_src_ids = unique_src_ids[unique_src_ids >= 0]

    # 6. Target Dataset（Open-Set 验证 & 无监督 DA）
    # tgt_idx was fixed by the mutually-exclusive benchmark split above.
    # Novel cells are retained regardless of ontology mapping; shared target cells
    # remain in the benchmark even if an ontology-mapping issue is later flagged.
    is_novel_gt    = np.isin(labels[tgt_idx], list(novel)).astype(np.float32)
    tgt_eval_valid_mask = src_valid_mask[tgt_idx]   # 仅供训练后 known-accuracy scoring

    # ── [S11] novel ratio 子采样 ────────────────────────────────────────────
    novel_ratio = getattr(args, 'novel_ratio', 1.0)
    if novel_ratio < 1.0 - 1e-6:
        rng_nr = np.random.RandomState(args.seed + 9999)   # 独立 seed，不干扰其他采样
        novel_pos  = np.where(is_novel_gt == 1)[0]         # 在 tgt_idx 内的 novel 位置
        known_pos  = np.where(is_novel_gt == 0)[0]
        n_keep     = max(1, int(len(novel_pos) * novel_ratio))
        kept_novel = rng_nr.choice(novel_pos, n_keep, replace=False)
        kept_all   = np.sort(np.concatenate([known_pos, kept_novel]))
        # 更新三个相关变量
        tgt_idx        = tgt_idx[kept_all]
        is_novel_gt    = is_novel_gt[kept_all]
        tgt_eval_valid_mask = src_valid_mask[tgt_idx]
        print(f"[Data] Novel ratio={novel_ratio:.1f}: "
              f"kept {n_keep}/{len(novel_pos)} novel cells "
              f"(total target: {len(tgt_idx)})")

    tgt_ds = TensorDataset(
        torch.tensor(X[tgt_idx],      dtype=torch.float),
        torch.tensor(raw_X[tgt_idx],  dtype=torch.float),
        torch.tensor(sf[tgt_idx],     dtype=torch.float),
        indices_global[tgt_idx],                                       # 本体 ID
        torch.tensor(tgt_eval_valid_mask, dtype=torch.bool),           # benchmark evaluation only
        torch.tensor(is_novel_gt,     dtype=torch.float),
        torch.tensor(raw_label_ids[tgt_idx], dtype=torch.long),              # GT novelty（验证用）
    )

    print(f"[Data] Source: {len(src_ds):>6} cells | "
          f"Target: {len(tgt_ds):>6} cells "
          f"(Novel GT: {int(is_novel_gt.sum())}, Known GT: {int((is_novel_gt == 0).sum())})")
    print(f"[Data] Unique src ontology IDs: {len(unique_src_ids)}")

    # 训练与标定使用不同的 source loader：标定必须覆盖全部 reference cells，
    # 且不能受 shuffle/drop_last 影响。
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

    # Benchmark GT 只存在于 eval dataset；训练 target dataset 仅包含无标签输入。
    # 复用 eval tensors 的底层存储，避免为大型 target 矩阵额外复制内存。
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

    # 防止后续维护中将 GT 字段重新暴露给 target training path。
    assert isinstance(tgt_train_loader.dataset, TensorDataset)
    assert len(tgt_train_loader.dataset.tensors) == 3, (
        "Target training dataset must contain exactly (x, raw_x, size_factor); "
        "benchmark labels are evaluation-only."
    )
    assert len(tgt_eval_loader.dataset.tensors) == 7

    novel_list = sorted(list(novel))   # 固定顺序，供 validate_osr 用

    return (src_train_loader, src_eval_loader, tgt_eval_loader, tgt_train_loader,
            X.shape[1], unique_src_ids, novel_list, le)


# =============================================================================
# 2. Decision-level Adversarial Loss（论文 4.3）
# =============================================================================

def compute_adv_loss(
    s_coarse_max: torch.Tensor,   # [B]，最大 coarse 余弦相似度（已 detach 自 prototype）
    logits_fine:  torch.Tensor,   # [B, N]，scaled 余弦 logit
    targets:      torch.Tensor,   # [B]，本体 ID（long）
    scale:        float = 8.0,    # model.scale 当前值（余弦→logit 换算用）
    beta:         float = 0.3,    # 要求的最小 fine-coarse gap（余弦空间）
    adv_margin:   float = 0.05,
    debug:        bool  = False,
    global_step:  int   = None,
    log_every:    int   = 200,
) -> torch.Tensor:
    """
    Source-domain Decision-level Causal Intervention（论文 §2.2 L_adv）

    设计意图（source 端因果干预）：
        已知细胞应以细粒度特异性特征做决策，而非仅依赖 coarse 谱系共性。
        训练已知细胞的决策模式，使 encoder 学到 fine-grained discriminability，
        间接保护 novel 细胞免于因共享谱系特征而被误归为已知类型（familiarity trap）。

    正确实现（修复 v1 的方向冲突）：
        ─────────────────────────────────────────────────────────────
        旧实现：ReLU(S_coarse - threshold)
            梯度 = +p_coarse → 推离 coarse prototype
            冲突：fine prototype 在 coarse 子空间内（by HCL），
                  推离 coarse 即推离 fine，与 L_proto / L_HCL 直接对抗。

        新实现：ReLU(beta - gap)，gap = fine_cos - S_coarse
            梯度 = -∂fine_cos/∂z = -p_fine[y]/scale
            optimizer descent: z += lr * p_fine[y]/scale → 推向 fine prototype ✓
            与 L_proto、L_HCL 同向，无冲突。
        ─────────────────────────────────────────────────────────────

    S_coarse 角色转变：
        旧：梯度目标（v1 试图改变 S_coarse 本身）
        新：只读门控信号（detach 后仅用于判断"gap 是否足够"）
            → Bug 2（coarse prototype 无独立梯度）变得无害：
              S_coarse 只需大致准确作为门控，不需要独立塑形。
              Warmup（epoch 50）保证 fine prototype 收敛后 coarse 均值已有意义。

    数学：
        fine_cos = logit_fine[true_class] / scale  ∈ [-1, 1]
        S_coarse_gate = s_coarse_max.detach()      ∈ [-1, 1]
        gap = fine_cos - S_coarse_gate
        L_adv = E[ I(argmax = y) * ReLU((beta - adv_margin) - gap) ]
    """
    if s_coarse_max.numel() == 0:
        return torch.tensor(0.0, device=s_coarse_max.device, requires_grad=True)

    # I(correct) 门控：仅对分类正确的源细胞激活
    correct_mask = (logits_fine.argmax(dim=1) == targets).float()   # [B]

    # 真实类别的余弦相似度（logit 除以 scale，还原到余弦空间）
    batch_ids = torch.arange(len(targets), device=targets.device)
    scale_safe = max(abs(scale), 1e-4)
    fine_cos = (logits_fine[batch_ids, targets] / scale_safe).clamp(-1.0, 1.0)  # [B]

    # S_coarse 作为只读门控（detach 确保不改变 prototype 几何）
    s_coarse_gate = s_coarse_max.detach()                            # [B]

    # fine-coarse gap：gap 小 = 模型仍依赖 coarse 谱系特征做决策
    gap = fine_cos - s_coarse_gate                                   # [B]

    # 惩罚 gap 不足（gap < beta）的正确分类源细胞
    threshold = beta - adv_margin
    loss_vec  = correct_mask * F.relu(threshold - gap)
    loss_adv  = loss_vec.mean()

    if debug and (global_step is None or global_step % log_every == 0):
        with torch.no_grad():
            g  = gap.detach()
            cm = correct_mask.detach()
            print(
                f"[ADV] step={global_step} "
                f"gap: min={g.min():.3f} mean={g.mean():.3f} max={g.max():.3f} | "
                f"threshold={threshold:.3f} | "
                f"correct={cm.mean():.3f} | "
                f"fire_ratio={(gap < threshold).float().mean():.3f}"
            )

    return loss_adv

def compute_target_adv_loss(
    z_tgt_norm: torch.Tensor,        # [B_t, d] L2-normalized
    s_coarse_max_tgt: torch.Tensor,  # [B_t]
    logits_fine_tgt: torch.Tensor,   # [B_t, N]
    p_norm: torch.Tensor,            # [N, d] L2-normalized
    alpha: float,                    # MLS calibration threshold
    beta: float = 0.3,
    adv_margin: float = 0.05,
    temp: float = 5.0,
    debug: bool = False,
    global_step: int = None,
    log_every: int = 100,
) -> torch.Tensor:
    """
    Target-domain Decision-level Adversarial Loss (论文 idea 3 的正确实现)
    
    设计：对高 MLS 的 target cell（模型自信"这是 known"的样本）
          施加 fine-coarse gap 约束，强迫其分类依赖 fine-specific 特征
          而非 coarse-shared 特征。
    
    与 novel_sep 互补：
      - novel_sep 推 likely-novel（MLS 低）远离 known prototypes
      - 此 loss 约束 likely-known（MLS 高）的决策必须 fine 主导
    """
    if z_tgt_norm.numel() == 0:
        return torch.zeros((), device=z_tgt_norm.device, requires_grad=True)
    
    mls = logits_fine_tgt.max(dim=1).values                # [B_t]
    pseudo_y = logits_fine_tgt.argmax(dim=1)               # [B_t]
    
    # 软门控：sigmoid((MLS - alpha) * temp)
    # MLS 越高 → 越像 known → conf_w 越接近 1 → 施加 gap 约束
    # MLS 越低 → 越像 novel → conf_w 越接近 0 → 放手不管（让 novel_sep 处理）
    conf_w = torch.sigmoid((mls - alpha) * temp).detach()  # [B_t]
    
    if conf_w.sum() < 1e-6:
        return torch.zeros((), device=z_tgt_norm.device, requires_grad=True)
    
    # gap = z 对 pseudo-known 类 fine prototype 的 cos - z 对 max coarse 的 cos
    fine_cos = (z_tgt_norm * p_norm[pseudo_y]).sum(dim=-1) # [B_t]
    s_coarse_gate = s_coarse_max_tgt.detach()              # [B_t]
    gap = fine_cos - s_coarse_gate                         # [B_t]
    
    threshold = beta - adv_margin
    loss_vec = conf_w * F.relu(threshold - gap)
    loss = loss_vec.sum() / (conf_w.sum() + 1e-8)
    
    if debug and (global_step is None or global_step % log_every == 0):
        with torch.no_grad():
            fire_rate = ((gap < threshold) & (conf_w > 0.5)).float().mean()
            print(
                f"[TGT_ADV] step={global_step} "
                f"alpha={alpha:.3f} | "
                f"MLS mean={mls.mean():.2f} | "
                f"conf_w mean={conf_w.mean():.3f} | "
                f"gap mean={gap.mean():.3f} | "
                f"fire_rate(conf>0.5 ∩ gap<thr)={fire_rate:.3f}"
            )
    
    return loss

def compute_target_neighborhood_consistency_loss(
    z_tgt: torch.Tensor,                    # [B, D], gradient target
    logits_fine: torch.Tensor,              # [B, N], detached gates only
    known_ids: torch.Tensor,                # [K] observed source ontology IDs
    ancestor_matrix: torch.Tensor,          # [N, N] static ontology ancestry
    depths: torch.Tensor,                   # [N] ontology depths
    *,
    alpha: float = None,                    # reference-calibrated only; None before calibration
    epoch: int = 1,
    k: int = 5,
    margin: float = 0.05,
    positive_floor: float = 0.80,
    pre_alpha_scale: float = 0.10,
    safe_k_multiplier: int = 2,
    negative_gap: float = 0.02,
    triplet_weight: float = 0.25,
    ontology_gate: str = 'soft',            # off | soft
    ontology_gate_strength: float = 0.25,
    ontology_temperature: float = 2.0,
    ontology_min_depth: int = 2,
    confidence_gate: bool = True,
    confidence_temperature: float = 5.0,
):
    """
    Conservative label-free target-neighbourhood consistency (TNC-v2.1).

    Protocol invariants
    -------------------
    * No target labels and no target/private class cardinality are accepted.
    * ``alpha`` is the existing reference-only calibrated MLS threshold.
    * All neighbour mining, ontology weighting and q_novel weighting are detached.
      Gradients from this objective therefore act on target ``z`` only.

    Why v2.1 differs from rejected TNC-v1
    -----------------------------------
    1. Only anchors with a genuine mutual-kNN positive are used; no top-1 fallback.
    2. Before alpha exists, TNC is a weak positive-preservation term only. There are
       no negatives in this phase.
    3. Positive pull stops once cosine similarity reaches ``positive_floor``; local
       neighbours are not contracted indefinitely.
    4. After alpha exists, pair weight is dominated by q_novel(i) q_novel(j), where
       q_novel is derived solely from reference-calibrated MLS. Known cells therefore
       receive little target-neighbour pressure.
    5. A negative is eligible only outside the top-(safe_k_multiplier*k) neighbourhood
       AND when it is already at least ``negative_gap`` farther than the positive.
       This explicitly avoids treating nearby non-mutual cells as hard negatives.
    6. Ontology information only softly ranks/weights positive pairs. It never creates
       a negative edge. Compatibility is based on expected deep-ancestor Jaccard
       similarity among the observed source ontology terms, avoiding the near-uniform
       coarse-vector cosine used by v1.
    """
    zero_stats = {
        'mutual_frac': 0.0, 'active_frac': 0.0, 'safe_neg_frac': 0.0,
        'pos_sim': 0.0, 'neg_sim': 0.0, 'pos_onto': 1.0, 'neg_onto': 1.0,
        'pair_weight': 0.0, 'q_novel': 0.0, 'pre_alpha': float(alpha is None),
    }
    if z_tgt.ndim != 2 or z_tgt.size(0) < 4 or known_ids.numel() == 0:
        return z_tgt.sum() * 0.0, zero_stats
    if ontology_gate not in {'off', 'soft'}:
        raise ValueError(f"Unsupported ontology_gate={ontology_gate!r}; expected 'off' or 'soft'")

    B = z_tgt.size(0)
    k_eff = min(max(int(k), 1), max(1, B - 2))
    safe_k = min(max(k_eff + 1, int(safe_k_multiplier) * k_eff), B - 1)
    z = F.normalize(z_tgt, p=2, dim=1)

    with torch.no_grad():
        z_det = z.detach()
        sim_det = z_det @ z_det.t()
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        sim_no_self = sim_det.masked_fill(eye, float('-inf'))

        # Reliable positive candidates: mutual kNN only, with no fallback.
        knn_idx = torch.topk(sim_no_self, k=k_eff, dim=1, largest=True).indices
        knn_mask = torch.zeros((B, B), dtype=torch.bool, device=z.device)
        knn_mask.scatter_(1, knn_idx, True)
        mutual = knn_mask & knn_mask.t()
        mutual.fill_diagonal_(False)
        has_mutual = mutual.any(dim=1)

        # Ontology compatibility among observed source terms.
        if ontology_gate == 'soft':
            src_logits = logits_fine.detach()[:, known_ids]
            src_prob = F.softmax(
                src_logits / max(float(ontology_temperature), 1e-6), dim=1
            )

            anc = ancestor_matrix[known_ids].float()                 # [K, N]
            depth_mask = depths >= int(ontology_min_depth)
            if depth_mask.any():
                anc = anc[:, depth_mask]
                # Jaccard relation between observed ontology terms using informative ancestors.
                inter = anc @ anc.t()
                cnt = anc.sum(dim=1, keepdim=True)
                union = cnt + cnt.t() - inter
                relation = torch.where(
                    union > 0, inter / union.clamp_min(1.0), torch.eye(
                        known_ids.numel(), device=z.device, dtype=anc.dtype
                    )
                ).clamp(0.0, 1.0)
                # Expected ontology compatibility between target cells; detached by construction.
                onto_pair = (src_prob @ relation) @ src_prob.t()
                onto_pair = onto_pair.clamp(0.0, 1.0)
            else:
                onto_pair = torch.ones_like(sim_det)
        else:
            onto_pair = torch.ones_like(sim_det)

        gate_s = min(max(float(ontology_gate_strength), 0.0), 1.0)

        # Positive selection inside mutual-kNN only.
        pos_rank = sim_det + gate_s * onto_pair
        pos_rank = pos_rank.masked_fill(~mutual, float('-inf'))
        pos_idx = pos_rank.argmax(dim=1)
        # Safe dummy index for anchors excluded by has_mutual; their weight is zero below.
        pos_idx = torch.where(has_mutual, pos_idx, torch.arange(B, device=z.device))
        rows = torch.arange(B, device=z.device)
        pos_sim_det = sim_det[rows, pos_idx]
        pos_onto = onto_pair[rows, pos_idx]

        # Safety neighbourhood: negatives may not come from the closest top-2k region.
        safe_idx = torch.topk(sim_no_self, k=safe_k, dim=1, largest=True).indices
        safe_mask = torch.zeros((B, B), dtype=torch.bool, device=z.device)
        safe_mask.scatter_(1, safe_idx, True)
        neg_valid = (~safe_mask) & (~eye)

        # Additional separation guard: candidate negative must already be clearly farther
        # than the selected positive, otherwise do not create a triplet for this anchor.
        neg_valid &= sim_det <= (pos_sim_det[:, None] - float(negative_gap))
        has_safe_neg = neg_valid.any(dim=1) & has_mutual
        neg_rank = sim_det.masked_fill(~neg_valid, float('-inf'))
        neg_idx = neg_rank.argmax(dim=1)
        neg_idx = torch.where(has_safe_neg, neg_idx, rows)
        neg_onto = onto_pair[rows, neg_idx]

        # Ontology is positive evidence only: it can strengthen a reliable positive,
        # but cannot turn any cell into a negative.
        if ontology_gate == 'soft':
            onto_weight = (1.0 - gate_s) + gate_s * pos_onto
        else:
            onto_weight = torch.ones(B, device=z.device)

        if alpha is None:
            q_novel = torch.zeros(B, device=z.device)
            pair_weight = has_mutual.float() * onto_weight * float(pre_alpha_scale)
        else:
            mls = logits_fine.detach().max(dim=1).values
            q_novel = torch.sigmoid(
                (float(alpha) - mls) * float(confidence_temperature)
            )
            if confidence_gate:
                q_pos = q_novel[pos_idx]
                pair_weight = has_mutual.float() * onto_weight * q_novel * q_pos
            else:
                pair_weight = has_mutual.float() * onto_weight

        pair_weight = pair_weight.clamp_min(0.0)

    rows = torch.arange(B, device=z.device)
    pos_sim = (z * z[pos_idx]).sum(dim=1)
    # Stop pulling once a mutual positive is already sufficiently close.
    pull = F.relu(float(positive_floor) - pos_sim)

    if alpha is None:
        triplet = torch.zeros_like(pull)
    else:
        neg_sim = (z * z[neg_idx]).sum(dim=1)
        triplet = F.relu(neg_sim - pos_sim + float(margin)) * has_safe_neg.float()

    loss_vec = pull + float(triplet_weight) * triplet

    # IMPORTANT: normalize by the number of structurally eligible anchors, not by
    # sum(pair_weight).  pair_weight contains pre-alpha scale, q_novel confidence
    # and ontology weights; dividing by its own sum would cancel their intended
    # absolute down-weighting and silently restore near-full TNC strength.
    eligible_count = has_mutual.float().sum().clamp_min(1.0)
    loss = (pair_weight * loss_vec).sum() / eligible_count

    with torch.no_grad():
        if alpha is None:
            neg_sim_stat = 0.0
        elif has_safe_neg.any():
            neg_sim_stat = float(neg_sim[has_safe_neg].mean().item())
        else:
            neg_sim_stat = 0.0
        active = ((pull > 0) | (triplet > 0)) & (pair_weight > 0)
        stats = {
            'mutual_frac': float(has_mutual.float().mean().item()),
            'active_frac': float(active.float().mean().item()),
            'safe_neg_frac': float(has_safe_neg.float().mean().item()) if alpha is not None else 0.0,
            'pos_sim': float(pos_sim[has_mutual].mean().item()) if has_mutual.any() else 0.0,
            'neg_sim': neg_sim_stat,
            'pos_onto': float(pos_onto[has_mutual].mean().item()) if has_mutual.any() else 1.0,
            'neg_onto': float(neg_onto[has_safe_neg].mean().item()) if has_safe_neg.any() else 1.0,
            'pair_weight': float(pair_weight.mean().item()),
            'q_novel': float(q_novel.mean().item()) if alpha is not None else 0.0,
            'pre_alpha': float(alpha is None),
        }
    return loss, stats

def compute_novel_sep_loss(
    z_tgt: torch.Tensor,              # [B, D]
    logits_fine: torch.Tensor,        # [B, N]
    prototypes: torch.Tensor,         # [N, D]
    known_ids: torch.Tensor,          # [n_known]
    alpha: float,
    temp: float = 5.0,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Label-free target novel-separation loss used after reference calibration.

    Cells with MLS below the frozen reference-derived threshold receive larger
    weights and are penalized when their maximum cosine similarity to observed
    source prototypes exceeds ``margin``. Target ground-truth labels are never
    used by this loss. In the paper-primary schedule it activates at epoch 60.
    """
    if z_tgt.numel() == 0 or known_ids.numel() == 0:
        return z_tgt.new_tensor(0.0, requires_grad=True)

    mls = logits_fine.max(dim=1).values                  # [B]
    novel_w = torch.sigmoid((alpha - mls) * temp).detach()   # [B], 越小MLS越像novel
    if novel_w.sum() < 1e-6:
        return z_tgt.new_tensor(0.0, requires_grad=True)

    z = F.normalize(z_tgt, p=2, dim=1)                   # [B, D]
    p_known = F.normalize(prototypes[known_ids], p=2, dim=1)  # [K, D]

    sim = torch.matmul(z, p_known.t())                   # [B, K]
    max_sim = sim.max(dim=1).values                      # [B]

    # 让 likely-novel 样本不要贴近任何 known prototype
    loss_vec = novel_w * F.relu(max_sim - margin)
    return loss_vec.sum() / (novel_w.sum() + 1e-8)

# =============================================================================
# 3. 推理阶段：MLS 阈值标定（论文 6.3）
# =============================================================================

CALIBRATION_SEED_OFFSET = 10_007


def build_calibration_plan(
    src_loader,
    *,
    base_seed: int,
    num_mask: int = 10,
    n_simulations: int = 20,
) -> dict:
    """
    预先生成并冻结 reference calibration 的 held-out class draws。

    该计划只依赖 reference labels，并使用与训练 RNG 隔离的局部 Generator。
    同一 run 的训练期 calibration、fixed-final evaluation 和全部 post-hoc
    snapshots 必须复用完全相同的 draws。
    """
    all_targets = []
    for _, _, _, y, _ in src_loader:
        all_targets.append(y.cpu())
    if not all_targets:
        raise RuntimeError("Cannot build calibration plan from an empty source loader")

    classes = torch.unique(torch.cat(all_targets)).sort().values
    if len(classes) < 2:
        raise RuntimeError("Reference calibration requires at least two source classes")

    n_to_mask = min(int(num_mask), len(classes) - 1)
    generator = torch.Generator(device='cpu')
    draw_seed = int(base_seed) + CALIBRATION_SEED_OFFSET
    generator.manual_seed(draw_seed)

    heldout_class_ids = []
    for _ in range(int(n_simulations)):
        perm = torch.randperm(len(classes), generator=generator)
        heldout_class_ids.append([
            int(v) for v in classes[perm[:n_to_mask]].tolist()
        ])

    return {
        'draw_seed': draw_seed,
        'base_seed': int(base_seed),
        'seed_offset': CALIBRATION_SEED_OFFSET,
        'num_mask_requested': int(num_mask),
        'num_mask_effective': int(n_to_mask),
        'n_simulations': int(n_simulations),
        'source_class_ids': [int(v) for v in classes.tolist()],
        'heldout_class_ids': heldout_class_ids,
    }


def calibrate_threshold(
    model,
    src_loader,
    device,
    epsilon: float = 0.05,
    *,
    calibration_plan: dict,
) -> float:
    """
    使用预先冻结的 reference-only held-out class draws 标定 MLS 阈值。

    calibration_plan 在训练开始前生成并持久化。该函数不采样随机数，保证
    不同 epochs/snapshots 的 alpha 差异只来自模型，而不是 calibration 噪声。
    """
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x, _, _, y, _ in src_loader:
            out = model(x.to(device))
            all_logits.append(out['logits_fine'].cpu())
            all_targets.append(y.cpu())

    if not all_logits:
        raise RuntimeError("Cannot calibrate threshold from an empty source loader")

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    mls = all_logits.max(dim=1).values
    classes = set(int(v) for v in torch.unique(all_targets).tolist())

    draws = calibration_plan.get('heldout_class_ids', [])
    if not draws:
        raise ValueError("calibration_plan contains no held-out class draws")

    alphas = []
    for draw in draws:
        draw_set = set(int(v) for v in draw)
        unknown = draw_set - classes
        if unknown:
            raise ValueError(
                f"Calibration draw contains classes absent from source loader: {sorted(unknown)}"
            )
        mask_cls = torch.tensor(sorted(draw_set), dtype=all_targets.dtype)
        keep_mask = ~torch.isin(all_targets, mask_cls)
        known_mls = mls[keep_mask].numpy()
        if len(known_mls) == 0:
            raise RuntimeError("A calibration draw removed all reference cells")
        alphas.append(float(np.percentile(known_mls, epsilon * 100)))

    alpha = float(np.median(alphas))
    print(
        f"[Calibrate] alpha={alpha:.4f} "
        f"(mask={calibration_plan['num_mask_effective']}, "
        f"sims={len(draws)}, epsilon={epsilon}, "
        f"draw_seed={calibration_plan['draw_seed']})"
    )
    return alpha

# =============================================================================
# 4. 验证与后处理（论文 5.3 & 6.1）
# =============================================================================

def lineage_consistency_check(
    novel_z:            torch.Tensor,      # [N_novel, D]，CPU
    novel_coarse_probs: torch.Tensor,      # [N_novel, N_all]，CPU，已投影至全本体空间
    ontology_manager:   OntologyManager,
    min_depth:          int   = 2,
    coverage_thres:     float = 0.4,
    leiden_resolution:  float = 1.0,
    cluster_labels:     np.ndarray = None,
    leiden_n_neighbors: int = 15,
    leiden_random_state: int = 0,
) -> dict:
    """
    论文 5.3：谱系一致性检查（Lineage Consistency Check, LCC）

    novel_coarse_probs 由调用方（validate_osr）正确投影后传入：
        softmax(fine_logits[:, src_ids]) @ ancestor_matrix[src_ids, :]
    本函数内部不做矩阵乘法，维度对齐责任在调用方。

    流程：
      1. Leiden 图聚类（graph-based，比 KMeans 对单细胞数据更鲁棒）
      2. 每簇平均祖先概率向量，depth >= min_depth 过滤浅层噪声节点
      3. top-k 覆盖率 >= coverage_thres -> Consistent；否则 Ambiguous
      4. top-k 中深度最深的节点作为 Primary Lineage

    Returns:
        dict: {cluster_id: {cells, label, score, top1_score, status}}
    """
    # 惰性构建反向映射（index -> name）
    if not hasattr(ontology_manager, 'id2name'):
        ontology_manager._build_id2name()

    z_np = novel_z.cpu().numpy()
    if z_np.shape[0] < 10:
        print("[LCC] Too few cells (< 10), skipping.")
        return {}

    if cluster_labels is None:
        print(f"[LCC] Common Leiden clustering {z_np.shape[0]} predicted-novel cells "
              f"(resolution={leiden_resolution}, kNN={leiden_n_neighbors})...")
        clusters = common_leiden(
            z_np, resolution=leiden_resolution,
            n_neighbors=leiden_n_neighbors,
            random_state=leiden_random_state,
        )
    else:
        clusters = np.asarray(cluster_labels, dtype=np.int64)
        if clusters.shape[0] != z_np.shape[0]:
            raise ValueError("LCC cluster_labels length does not match novel_z")
        print(f"[LCC] Reusing Track-2 common Leiden partition: "
              f"{len(np.unique(clusters))} clusters.")

    depths_cpu   = ontology_manager.depths.cpu()
    coarse_probs = novel_coarse_probs   # [N_novel, N_all]，CPU tensor

    results = {}
    for cid in np.unique(clusters):
        mask_np = (clusters == cid)
        n_cells = int(mask_np.sum())
        if n_cells < 5:
            continue

        avg_probs      = coarse_probs[mask_np].mean(dim=0)            # [N_all]
        depth_mask     = (depths_cpu >= min_depth).float()
        filtered_probs = avg_probs.cpu() * depth_mask                 # [N_all]

        n_valid = int((filtered_probs > 0).sum().item())
        if n_valid == 0:
            continue

        top_k = 3
        topk_scores, topk_indices = torch.topk(filtered_probs, min(top_k, n_valid))
        top1_score = topk_scores[0].item()
        coverage   = topk_scores.sum().item() / (filtered_probs.sum().item() + 1e-8)

        if coverage >= coverage_thres:
            topk_depths  = depths_cpu[topk_indices].numpy()
            deepest_pos  = int(np.argmax(topk_depths))
            # lineage_consistency_check 里，Consistent 分支的 label 构建改成：
            deepest_idx  = topk_indices[deepest_pos].item()

            # 从 all_terms 取原始 CL ID
            deepest_cl_id = ontology_manager.all_terms[deepest_idx] \
                if deepest_idx < len(ontology_manager.all_terms) else f"idx_{deepest_idx}"

            # 从 id2name 取人类可读名称
            deepest_name = ontology_manager.id2name.get(deepest_idx, deepest_cl_id)

            top_names = []
            for i in topk_indices:
                i = i.item()
                cl_id = ontology_manager.all_terms[i] if i < len(ontology_manager.all_terms) else f"idx_{i}"
                readable = ontology_manager.id2name.get(i, cl_id)
                top_names.append(readable)

            label = (f"Potential subtype of <{deepest_name}> "
                    f"(in {' / '.join(top_names)} lineage)")
            status = "Consistent"
        else:
            label  = "Unknown / Mixed Lineage"
            status = "Ambiguous"

        results[cid] = {
            'cells':      n_cells,
            'label':      label,
            'score':      float(coverage),
            'top1_score': float(top1_score),
            'status':     status,
        }

    return results



def _print_lcc_report(report: dict):
    """格式化打印 LCC 结果表格。"""
    if not report:
        print("[LCC] No clusters with >= 5 cells found.")
        return

    print("\n" + "=" * 110)
    print(f"{'CID':<6} | {'Cells':>6} | {'Status':<12} | {'TopK-Cov':>9} | "
          f"{'Top1':>6} | Interpretation")
    print("-" * 110)
    for cid, info in sorted(report.items()):
        print(f"{cid:<6} | {info['cells']:>6} | {info['status']:<12} | "
              f"{info['score']:>9.3f} | {info['top1_score']:>6.3f} | {info['label']}")
    print("-" * 110)
    consistent = [v for v in report.values() if v['status'] == 'Consistent']
    avg_cov    = float(np.mean([v['score'] for v in report.values()]))
    print(f"[LCC Summary] Clusters={len(report)} | "
          f"Consistent={len(consistent)} | Avg TopK Coverage={avg_cov:.3f}")
    print("=" * 110 + "\n")


def validate_osr(
    model,
    tgt_loader,
    alpha:            float,
    device,
    ontology_manager: OntologyManager,
    unique_src_ids:   torch.Tensor,
    epoch,                             # int 或 "FINAL"
    le,
    novel_list:       list = None, 
    leiden_res:       float = 1.0,
    lcc_temperature=2.0,
    eval_kmeans_seed: int = 0,
    eval_kmeans_n_init: int = 10,
    leiden_n_neighbors: int = 15,
    leiden_random_state: int = 0,
    run_deployment_eval: bool = True,
    run_lcc_eval: bool = True,
    eval_details: dict = None,
) -> tuple:
    """
    Fixed-checkpoint post-training evaluator for OSR, discovery and LCC.

    Track 1 (benchmark/oracle-K): GT-novel cells only -> row-wise L2
    normalization -> KMeans(K_novel) with a frozen seed/n_init -> Hungarian
    scoring. Ground-truth labels determine K and scoring only; they never select
    KMeans restarts or training checkpoints.

    Track 2 (deployment): cells declared novel by MLS < alpha -> row-wise L2
    normalization -> fixed kNN graph -> Leiden. No ground-truth K is used to
    form this partition. The same Leiden labels are reused by LCC so discovery
    and lineage interpretation operate on one frozen partition.

    AUROC uses is_novel_gt only after training. Known accuracy is computed on
    benchmark-valid known cells. LCC projection uses
    softmax(fine_logits[:, src_ids]) @ ancestor_matrix[src_ids, :].
    """
    model.eval()

    all_mls            = []
    all_is_novel_gt    = []
    novel_z_cache        = []
    novel_logits_cache   = []
    novel_true_cache     = []
    novel_is_gt_cache    = []

    all_known_pred = []
    all_known_true = []
    all_novel_z_eval = []
    all_novel_true = []

    with torch.no_grad():
        for x, _, _, y_idx, valid_mask, is_novel_gt, y_raw in tgt_loader:
            x = x.to(device)

            out = model(x)

            logits = out['logits_fine'].cpu()
            z      = out['z'].cpu()

            y_idx       = y_idx.cpu()
            valid_mask  = valid_mask.cpu()
            is_novel_gt = is_novel_gt.cpu()
            y_raw       = y_raw.cpu()

            mls, _ = logits.max(dim=1)

            all_mls.append(mls)
            all_is_novel_gt.append(is_novel_gt)

            # ===== known accuracy =====
            known_mask_b = (is_novel_gt == 0) & valid_mask

            if known_mask_b.any():
                src_ids_cpu = unique_src_ids.cpu()

                logits_src = logits[:, src_ids_cpu]   # CPU
                pred_src   = logits_src.argmax(dim=1) # CPU

                true_src = torch.full_like(pred_src, -1)

                for k, gid in enumerate(src_ids_cpu):
                    true_src[y_idx == gid] = k

                mask = known_mask_b & (true_src >= 0)

                if mask.any():
                    all_known_pred.append(pred_src[mask])
                    all_known_true.append(true_src[mask])

            # novel accuracy 收集
            novel_mask_b = (is_novel_gt == 1)
            if novel_mask_b.any():
                all_novel_z_eval.append(z[novel_mask_b])       # ← 用已转好的 z
                all_novel_true.append(y_raw[novel_mask_b])

            # LCC 收集
            is_pred_novel = (mls < alpha)
            if is_pred_novel.any():
                novel_z_cache.append(z[is_pred_novel])
                novel_logits_cache.append(logits[is_pred_novel])
                novel_true_cache.append(y_raw[is_pred_novel])
                novel_is_gt_cache.append(is_novel_gt[is_pred_novel])
    
    all_mls         = torch.cat(all_mls)
    all_is_novel_gt = torch.cat(all_is_novel_gt)

    # AUROC
    n_novel = int(all_is_novel_gt.sum().item())
    n_known = int((all_is_novel_gt == 0).sum().item())
    if n_novel > 0 and n_known > 0:
        try:
            auroc = roc_auc_score(all_is_novel_gt.numpy(), -all_mls.numpy())
        except Exception as e:
            print(f"[Val] AUROC failed: {e}")
            auroc = 0.0
    else:
        auroc = 0.5
        print(f"[Val] WARNING: AUROC=0.5 — GT Novel={n_novel}, GT Known={n_known}. "
              f"Check ratio_shared / ratio_src_priv args.")

    n_pred_novel = sum(t.shape[0] for t in novel_z_cache) if novel_z_cache else 0
    
    # ── Known Accuracy ───────────────────────────────────────────────
    if all_known_pred:
        kp = torch.cat(all_known_pred).numpy()
        kt = torch.cat(all_known_true).numpy()
        known_acc = (kp == kt).mean() 
    else:
        known_acc = 0.0

    # ── Track 1: Common oracle-K KMeans (Table-1-compatible ruler) ───
    track1 = None
    nt_dense = None
    novel_pred = None
    if all_novel_z_eval and novel_list and len(novel_list) > 1:
        nz_t = torch.cat(all_novel_z_eval)
        nt = torch.cat(all_novel_true).numpy()

        uniq = np.unique(nt)
        remap = {v: i for i, v in enumerate(uniq)}
        nt_dense = np.array([remap[v] for v in nt], dtype=np.int64)
        n_clusters = len(uniq)

        track1 = common_oracle_kmeans(
            nz_t,
            nt_dense,
            n_clusters=n_clusters,
            random_state=eval_kmeans_seed,
            n_init=eval_kmeans_n_init,
        )
        novel_pred = track1['pred']
        novel_acc = track1['weighted_acc']
        novel_acc_macro = track1['macro_acc']
        novel_cm = track1['confusion']
        per_class_acc = novel_cm.diagonal() / np.maximum(novel_cm.sum(axis=1), 1)

        class_names = le.inverse_transform(uniq)
        print("\n[Track 1 | Common oracle-K KMeans]")
        print("  Input: GT-novel cells only -> L2 normalize -> KMeans(K_novel) -> Hungarian")
        print(f"  Frozen KMeans: seed={eval_kmeans_seed}, n_init={eval_kmeans_n_init}; "
              "no GT-based restart selection")
        print("  [Novel per-class accuracy]")
        for i, cls_name in enumerate(class_names):
            print(f"    {cls_name[:40]:<40}: {per_class_acc[i]*100:.1f}%")
        print(f"  Novel weighted/Hungarian : {novel_acc*100:.1f}%")
        print(f"  Novel macro              : {novel_acc_macro*100:.1f}%")
        print(f"  ARI/AMI/NMI              : {track1['ari']:.4f}/{track1['ami']:.4f}/{track1['nmi']:.4f}")
        print(f"  K_novel                  : {track1['k_novel']}")

        overall_j = compute_joint_overall(
            all_known_pred, all_known_true,
            novel_pred, nt_dense,
            n_known_classes=len(unique_src_ids),
        )
        track1['overall_joint'] = float(overall_j)
        track1['n_gt_novel_cells'] = int(len(nt_dense))
        print(f"  Overall joint Hungarian  : {overall_j*100:.1f}%")
    else:
        novel_acc = 0.0
        novel_acc_macro = 0.0
        overall_j = 0.0

    n_k = len(torch.cat(all_known_true)) if all_known_true else 0
    n_n = int(all_is_novel_gt.sum().item())
    overall_w = (n_k * known_acc + n_n * novel_acc_macro) / max(n_k + n_n, 1)
    n_pred_novel = sum(t.shape[0] for t in novel_z_cache) if novel_z_cache else 0

    # ── Track 2: Common Leiden / deployment (no K) ──────────────────
    track2 = None
    deployment_clusters = None
    if run_deployment_eval and novel_z_cache and n_pred_novel >= 10:
        pred_novel_z = torch.cat(novel_z_cache)
        pred_true = torch.cat(novel_true_cache).numpy()
        pred_is_gt = torch.cat(novel_is_gt_cache).numpy().astype(bool)
        deployment_clusters = common_leiden(
            pred_novel_z,
            resolution=leiden_res,
            n_neighbors=leiden_n_neighbors,
            random_state=leiden_random_state,
        )

        tp = int(pred_is_gt.sum())
        fp = int((~pred_is_gt).sum())
        fn = int(n_novel - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12))
        all_scores = _safe_cluster_scores(pred_true, deployment_clusters)
        tp_scores = _safe_cluster_scores(
            pred_true[pred_is_gt], deployment_clusters[pred_is_gt]
        )
        track2 = {
            'predicted_novel_cells': int(n_pred_novel),
            'gt_novel_cells': int(n_novel),
            'true_positive_novel_cells': tp,
            'false_positive_known_cells': fp,
            'false_negative_novel_cells': fn,
            'novel_detection_precision': float(precision),
            'novel_detection_recall': float(recall),
            'novel_detection_f1': float(f1),
            'leiden_clusters': int(len(np.unique(deployment_clusters))),
            'ari_all_predicted_novel': all_scores['ari'],
            'ami_all_predicted_novel': all_scores['ami'],
            'nmi_all_predicted_novel': all_scores['nmi'],
            'ari_tp_only': tp_scores['ari'],
            'ami_tp_only': tp_scores['ami'],
            'nmi_tp_only': tp_scores['nmi'],
            'leiden_resolution': float(leiden_res),
            'leiden_n_neighbors': int(leiden_n_neighbors),
            'leiden_random_state': int(leiden_random_state),
            'l2_normalized': True,
            'uses_ground_truth_k': False,
        }
        def _fmt(v):
            return 'NA' if v is None else f'{v:.4f}'
        print("\n[Track 2 | Common Leiden / deployment]")
        print("  Input: model-declared novel cells -> L2 normalize -> fixed Leiden; no K")
        print(f"  PredNovel={n_pred_novel} | TP={tp} FP={fp} FN={fn} | "
              f"P/R/F1={precision:.4f}/{recall:.4f}/{f1:.4f}")
        print(f"  Leiden clusters={track2['leiden_clusters']} | "
              f"ALL ARI/AMI/NMI={_fmt(all_scores['ari'])}/{_fmt(all_scores['ami'])}/{_fmt(all_scores['nmi'])} | "
              f"TP-only ARI/AMI/NMI={_fmt(tp_scores['ari'])}/{_fmt(tp_scores['ami'])}/{_fmt(tp_scores['nmi'])}")

    if eval_details is not None:
        eval_details.clear()
        eval_details.update({
            'evaluator_protocol': {
                'track1': 'GT-novel only -> L2 -> fixed KMeans(K_novel) -> Hungarian',
                'track2': 'predicted-novel -> L2 -> fixed Leiden, no K',
                'checkpoint_selection_uses_target_gt': False,
            },
            'track1_oracle_kmeans': None if track1 is None else {
                k: v for k, v in track1.items() if k not in {'pred', 'confusion'}
            },
            'track2_deployment_leiden': track2,
        })

    print(f"\n[Val | Epoch {epoch}] "
          f"known={known_acc*100:.1f} | novel={novel_acc*100:.1f} | "
          f"overall(w)={overall_w*100:.1f} | overall(joint)={overall_j*100:.1f} | "
          f"AUROC={auroc:.4f}  alpha={alpha:.4f}  "
          f"GT Novel={n_novel}  Pred Novel={n_pred_novel}")    

    # LCC（论文 5.3）
    run_lcc = run_lcc_eval and (
        (epoch == "FINAL") or (isinstance(epoch, int) and epoch % 20 == 0)
    )

    if run_lcc and novel_z_cache:
        lcc_novel_z           = torch.cat(novel_z_cache)        # [N_pred, D]
        all_novel_fine_logits = torch.cat(novel_logits_cache)   # [N_pred, N_onto]

        if lcc_novel_z.shape[0] >= 10:
            compute_device = ontology_manager.ancestor_matrix.device
            src_ids_dev    = unique_src_ids.to(compute_device)  # [n_src]

            # [F3] 先切列（src 列），再乘祖先矩阵
            fine_logits_src = all_novel_fine_logits.to(compute_device)[:, src_ids_dev]
            # [N_pred, n_src]

            anc_mat = ontology_manager.ancestor_matrix.float()[src_ids_dev, :]
            # [n_src, N_all]

            fine_probs = F.softmax(fine_logits_src / lcc_temperature, dim=1)
            novel_coarse_probs = torch.mm(fine_probs, anc_mat).cpu()                                  # [N_pred, N_all]

            print(f"[LCC] Running on {lcc_novel_z.shape[0]} predicted novel cells...")
            try:
                report = lineage_consistency_check(
                    lcc_novel_z,
                    novel_coarse_probs,
                    ontology_manager,
                    min_depth         = 2,
                    coverage_thres    = 0.4,
                    leiden_resolution = leiden_res,
                    cluster_labels     = deployment_clusters,
                    leiden_n_neighbors = leiden_n_neighbors,
                    leiden_random_state = leiden_random_state,
                )
                _print_lcc_report(report)
                # ── LCC 阈值敏感性分析 ────────────────────────────────────────
                if report:
                    print("[LCC Threshold Sensitivity]")
                    for thres in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
                        n_consistent = sum(1 for v in report.values()
                                        if v['score'] >= thres)
                        print(f"  coverage_thres={thres:.2f}: "
                            f"Consistent={n_consistent}/{len(report)}")
            except Exception as e:
                print(f"[LCC] Failed: {e}")
        else:
            print(f"[LCC] Skipped: {lcc_novel_z.shape[0]} predicted novel cells < 10.")

    return auroc, overall_w, overall_j, known_acc, novel_acc, novel_acc_macro



# =============================================================================
# 5. Checkpoint 工具
# =============================================================================

def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    args,
    *,
    role: str,
    selection_rule: str,
    calibration_plan: dict,
    calibration_plan_path: str,
):
    """
    保存预先规定的 checkpoint。checkpoint 元数据明确记录：
    target benchmark metrics 未参与学习率、早停或 checkpoint selection。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_only_provenance = getattr(args, '_source_only_hvg_provenance', None)
    torch.save({
        'epoch': epoch,
        'checkpoint_role': role,
        'selection_rule': selection_rule,
        'target_metrics_used_for_selection': False,
        'target_training_labels_present': False,
        'benchmark_target_cells_present': (
            False if args.protocol_profile == 'source_only_hvg' else None
        ),
        'source_only_hvg_provenance': source_only_provenance,
        'learning_rate_schedule': {
            'name': 'cosine_annealing_shared_horizon',
            'base_lr': float(args.lr),
            'min_lr': float(args.min_lr),
            'schedule_epochs': int(args.lr_schedule_epochs),
            'run_epochs': int(args.epochs),
        },
        'calibration_plan': calibration_plan,
        'protocol_schedule': _schedule_summary(args),
        'calibration_plan_path': str(calibration_plan_path),
        # 保留旧键以兼容已有读取脚本，但 primary checkpoint 中不写入 target 指标。
        'auroc': None,
        'overall_acc': None,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'sched_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'args': vars(args),
    }, path)
    print(f"[Checkpoint] Saved {role}: '{path}' "
          f"(epoch={epoch}, selection={selection_rule})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """加载 checkpoint；optimizer/scheduler 仅在续训时恢复。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optim_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optim_state_dict'])
    if scheduler is not None and ckpt.get('sched_state_dict') is not None:
        scheduler.load_state_dict(ckpt['sched_state_dict'])
    print(f"[Checkpoint] Loaded '{path}' "
          f"(epoch={ckpt.get('epoch')}, role={ckpt.get('checkpoint_role', 'legacy')}, "
          f"selection={ckpt.get('selection_rule', 'legacy')})")
    # 保持旧函数的返回契约；fixed-final checkpoint 的 AUROC 为 None。
    return ckpt.get('epoch'), ckpt.get('auroc')


def _snapshot_checkpoint_path(final_path: str, epoch: int) -> str:
    """从 final checkpoint 路径派生固定 epoch snapshot 路径。"""
    path = Path(final_path)
    suffix = path.suffix or '.pth'
    return str(path.with_name(f"{path.stem}_epoch{epoch:03d}{suffix}"))


def _cosine_learning_rate(
    *,
    epoch: int,
    schedule_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Prespecified epoch-wise cosine schedule on a shared absolute horizon.

    All primary, ablation and cross-dataset runs use the same LR at the same
    absolute epoch. With the default 100-epoch horizon, a 60-epoch cross-dataset
    run uses the first 60 steps rather than compressing the whole cosine curve
    into 60 epochs.

        lr(e) = min_lr + 0.5 * (base_lr - min_lr)
                * (1 + cos(pi * (e - 1) / (H - 1))).

    Epochs beyond H remain at min_lr.
    """
    if schedule_epochs <= 1:
        return float(min_lr)
    clipped_epoch = min(max(int(epoch), 1), int(schedule_epochs))
    progress = (clipped_epoch - 1) / (int(schedule_epochs) - 1)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = float(lr)


def _default_snapshot_epochs(args) -> set[int]:
    """Derive prespecified snapshots from the declared training budget, never target metrics."""
    if args.protocol_profile == 'source_only_hvg':
        # Development screen output is fixed-final only.
        return set()
    if args.snapshot_epochs is not None:
        return {int(e) for e in args.snapshot_epochs if 1 <= int(e) < args.epochs}

    if getattr(args, 'cross_dataset', False):
        # Cross-dataset primary runs conventionally use 60 epochs; retain the two
        # immediately preceding 10-epoch trajectory points (e.g. 40/50/60).
        return {
            e for e in (args.epochs - 20, args.epochs - 10)
            if 1 <= e < args.epochs
        }

    # Intra-dataset snapshots begin at the first calibration epoch and then follow
    # the fixed calibration interval (e.g. 60/70/80/90/100).
    return {
        e for e in range(args.first_calibrate_epoch, args.epochs, args.calibrate_every)
        if 1 <= e < args.epochs
    }



def _validate_protocol_configuration(args) -> None:
    """
    Validate schedule invariants and, when requested, the manuscript protocol.

    protocol_profile:
      - paper_primary: enforce the Table S4 / headline benchmark defaults.
      - paper_schedule: enforce the same optimisation schedule but allow a
        prespecified robustness perturbation such as labelled ratio or novel ratio.
      - source_only_hvg: enforce the frozen source-derived 2k/4k development episode.
      - custom: retain only safety/invariant checks.
    """
    if args.epochs < 1:
        raise ValueError('--epochs must be >= 1')
    if args.batch_size < 1:
        raise ValueError('--batch_size must be >= 1')
    if not (0.0 < args.source_label_ratio <= 1.0):
        raise ValueError('--source_label_ratio must lie in (0, 1]')
    if not (0.0 <= args.rec_decay_rate <= 1.0):
        raise ValueError('--rec_decay_rate must lie in [0, 1]')
    if not (0.0 <= args.rec_floor <= 1.0):
        raise ValueError('--rec_floor must lie in [0, 1]')
    if args.rec_rampdown_epochs < 1:
        raise ValueError('--rec_rampdown_epochs must be >= 1')
    if not (1 <= args.hcl_warmup_epoch < args.adv_warmup_epoch):
        raise ValueError('Expected 1 <= hcl_warmup_epoch < adv_warmup_epoch')
    if args.first_calibrate_epoch < args.adv_warmup_epoch:
        raise ValueError('--first_calibrate_epoch cannot precede --adv_warmup_epoch')
    if args.novel_sep_start_epoch < args.first_calibrate_epoch:
        raise ValueError('--novel_sep_start_epoch cannot precede --first_calibrate_epoch')
    if args.rec_stop_epoch < args.adv_warmup_epoch:
        raise ValueError('--rec_stop_epoch cannot precede --adv_warmup_epoch')
    if args.lr_schedule_epochs < args.epochs:
        raise ValueError(
            '--lr_schedule_epochs must be >= --epochs so the run does not pass '
            'the declared cosine horizon'
        )
    if not (0.0 <= args.min_lr <= args.lr):
        raise ValueError('--min_lr must satisfy 0 <= min_lr <= lr')
    if args.cross_standardization not in {'joint', 'per_dataset'}:
        raise ValueError('Unsupported --cross_standardization mode')
    if args.eval_kmeans_n_init < 1:
        raise ValueError('--eval_kmeans_n_init must be >= 1')
    if args.leiden_n_neighbors < 2:
        raise ValueError('--leiden_n_neighbors must be >= 2')
    if args.lambda_tnc < 0:
        raise ValueError('--lambda_tnc must be >= 0')
    if args.tnc_start_epoch < 1:
        raise ValueError('--tnc_start_epoch must be >= 1')
    if args.tnc_k < 1:
        raise ValueError('--tnc_k must be >= 1')
    if args.tnc_margin < 0:
        raise ValueError('--tnc_margin must be >= 0')
    if not (0.0 <= args.tnc_ontology_gate_strength <= 1.0):
        raise ValueError('--tnc_ontology_gate_strength must lie in [0, 1]')
    if args.tnc_ontology_temperature <= 0:
        raise ValueError('--tnc_ontology_temperature must be > 0')
    if args.tnc_confidence_temperature <= 0:
        raise ValueError('--tnc_confidence_temperature must be > 0')
    if not (0.0 <= args.tnc_positive_floor <= 1.0):
        raise ValueError('--tnc_positive_floor must lie in [0,1]')
    if not (0.0 <= args.tnc_pre_alpha_scale <= 1.0):
        raise ValueError('--tnc_pre_alpha_scale must lie in [0,1]')
    if args.tnc_safe_k_multiplier < 2:
        raise ValueError('--tnc_safe_k_multiplier must be >= 2')
    if args.tnc_negative_gap < 0:
        raise ValueError('--tnc_negative_gap must be >= 0')
    if args.tnc_triplet_weight < 0:
        raise ValueError('--tnc_triplet_weight must be >= 0')
    if args.tnc_ontology_min_depth < 0:
        raise ValueError('--tnc_ontology_min_depth must be >= 0')

    is_source_only = args.protocol_profile == 'source_only_hvg'
    if is_source_only:
        if args.dataset not in SOURCE_ONLY_HVG_DATASETS:
            raise ValueError(
                f"source_only_hvg --dataset must be one of {SOURCE_ONLY_HVG_DATASETS}"
            )
        if args.seed not in SOURCE_ONLY_HVG_ALLOWED_SEEDS:
            raise ValueError(
                f"source_only_hvg --seed must be one of {SOURCE_ONLY_HVG_ALLOWED_SEEDS}"
            )
        if args.source_only_fold not in SOURCE_ONLY_HVG_ALLOWED_FOLDS:
            raise ValueError(
                f"source_only_hvg --source_only_fold must be one of "
                f"{SOURCE_ONLY_HVG_ALLOWED_FOLDS}"
            )
        if args.hvg not in SOURCE_ONLY_HVG_ALLOWED_HVG:
            raise ValueError(
                f"source_only_hvg --hvg must be one of {SOURCE_ONLY_HVG_ALLOWED_HVG}"
            )
        if not args.source_only_manifest:
            raise ValueError('source_only_hvg requires --source_only_manifest')
        if args.cross_dataset:
            raise ValueError('source_only_hvg cannot be combined with --cross_dataset')
        if args.snapshot_epochs is not None:
            raise ValueError('source_only_hvg is fixed-final only; do not set --snapshot_epochs')
        if args.posthoc_oracle_diagnostic:
            raise ValueError('source_only_hvg forbids post-hoc target-selected checkpoint diagnostics')
        if args.n_cs_override is not None or args.n_cr_override is not None:
            raise ValueError('source_only_hvg forbids class-registry overrides')
        if not args.use_zinb:
            raise ValueError('source_only_hvg inherits paper-primary use_zinb=True')
    elif (args.source_only_manifest is not None
          or args.source_only_fold is not None
          or args.source_only_episode_probe_only):
        raise ValueError(
            'source-only manifest/fold/probe arguments require '
            '--protocol_profile source_only_hvg'
        )

    if args.protocol_profile == 'custom':
        return

    expected_schedule = {
        'hvg': args.hvg if is_source_only else 2000,
        'batch_size': 1024,
        'rec_weight': 1.0,
        'rec_decay_rate': 0.5,
        'rec_stop_epoch': 80,
        'rec_rampdown_epochs': 20,
        'rec_floor': 0.0,
        'hcl_warmup_epoch': 20,
        'adv_warmup_epoch': 50,
        'hcl_margin_start': 0.3,
        'hcl_margin_end': 0.05,
        'first_calibrate_epoch': 60,
        'novel_sep_start_epoch': 60,
        'lr': 1e-3,
        'min_lr': 1e-5,
        'lr_schedule_epochs': 100,
        'lambda_hcl': 0.1,
        'lambda_target_hcl': 0.0,
        'lambda_adv': 0.05,
        'lambda_novel_sep': 0.01,
        'adv_mode': 'source',
        'hcl_decay_start': 999,
        'adv_stop_epoch': 999,
        'calibrate_epsilon': 0.05,
        'calibration_num_mask': 10,
        'calibration_simulations': 20,
        'calibrate_every': 10,
    }
    mismatches = []
    for name, expected in expected_schedule.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matched = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            matched = actual == expected
        if not matched:
            mismatches.append(f'{name}={actual!r} (expected {expected!r})')

    expected_epochs = 60 if args.cross_dataset else 100
    if args.epochs != expected_epochs:
        mismatches.append(f'epochs={args.epochs!r} (expected {expected_epochs!r})')
    if args.cross_dataset and args.cross_standardization != 'joint':
        mismatches.append(
            f'cross_standardization={args.cross_standardization!r} '
            "(paper protocol requires 'joint')"
        )

    if args.protocol_profile in {
        'paper_primary', 'source_only_hvg', 'dev_tnc', 'dev_tnc_v2', 'dev_tnc_v2_zero'
    }:
        primary_expected = {
            'source_label_ratio': 0.5,
            'novel_ratio': 1.0,
        }
        for name, expected in primary_expected.items():
            actual = getattr(args, name)
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                mismatches.append(f'{name}={actual!r} (expected {expected!r})')

    if is_source_only:
        source_only_expected = {
            'beta': 0.3,
            'adv_margin': 0.05,
            'novel_sep_temp': 5.0,
            'novel_sep_margin': 0.1,
            'coarse_depth_threshold': 3,
            'eval_kmeans_seed': 0,
            'eval_kmeans_n_init': 10,
        }
        for name, expected in source_only_expected.items():
            actual = getattr(args, name)
            if isinstance(expected, float):
                matched = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matched = actual == expected
            if not matched:
                mismatches.append(f'{name}={actual!r} (expected {expected!r})')

    if args.protocol_profile in {'paper_primary', 'paper_schedule', 'source_only_hvg'}:
        if not math.isclose(float(args.lambda_tnc), 0.0, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(
                f'lambda_tnc={args.lambda_tnc!r} (paper baseline requires 0.0)'
            )

    if args.protocol_profile == 'dev_tnc':
        if args.lambda_tnc <= 0:
            mismatches.append('dev_tnc requires lambda_tnc > 0')
        if args.tnc_start_epoch != args.hcl_warmup_epoch:
            mismatches.append(
                f'tnc_start_epoch={args.tnc_start_epoch!r} '
                f'(Dev-B requires HCL-aligned start={args.hcl_warmup_epoch!r})'
            )
        if args.lambda_target_hcl != 0.0:
            mismatches.append('Dev-B keeps legacy target-HCL disabled (lambda_target_hcl=0)')
        if args.hcl_decay_start < args.epochs:
            mismatches.append('Dev-B keeps HCL decay disabled; use Dev-C only after TNC is diagnosed')

    if args.protocol_profile == 'dev_tnc_v2':
        if args.lambda_tnc <= 0:
            mismatches.append('dev_tnc_v2 requires lambda_tnc > 0')
        if args.tnc_start_epoch != args.hcl_warmup_epoch:
            mismatches.append(
                f'tnc_start_epoch={args.tnc_start_epoch!r} '
                f'(TNC-v2 requires HCL-aligned start={args.hcl_warmup_epoch!r})'
            )
        if args.lambda_target_hcl != 0.0:
            mismatches.append('TNC-v2 keeps legacy target-HCL disabled (lambda_target_hcl=0)')
        if args.hcl_decay_start < args.epochs:
            mismatches.append('TNC-v2 keeps HCL decay disabled')
        if not args.tnc_confidence_gate:
            mismatches.append('TNC-v2 requires reference-alpha q_novel gating after calibration')

    if args.protocol_profile == 'dev_tnc_v2_zero':
        if not math.isclose(float(args.lambda_tnc), 0.0, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append('dev_tnc_v2_zero requires lambda_tnc=0')
        if args.lambda_target_hcl != 0.0:
            mismatches.append('zero-TNC control keeps legacy target-HCL disabled')
        if args.hcl_decay_start < args.epochs:
            mismatches.append('zero-TNC control keeps HCL decay disabled')

    if mismatches:
        raise ValueError(
            f"Protocol profile '{args.protocol_profile}' mismatch:\n  - "
            + "\n  - ".join(mismatches)
            + "\nUse --protocol_profile custom only for a deliberately documented deviation."
        )


def _schedule_summary(args) -> dict:
    """Machine-readable schedule summary written to logs/checkpoints."""
    summary = {
        'protocol_profile': args.protocol_profile,
        'run_epochs': int(args.epochs),
        'lr': {
            'name': 'cosine_annealing_shared_horizon',
            'base': float(args.lr),
            'minimum': float(args.min_lr),
            'horizon_epochs': int(args.lr_schedule_epochs),
        },
        'reconstruction_weight': {
            'initial_weight': float(args.rec_weight),
            'initial_until_epoch': int(args.adv_warmup_epoch - 1),
            'mid_weight': float(args.rec_weight * args.rec_decay_rate),
            'mid_from_epoch': int(args.adv_warmup_epoch),
            'mid_until_epoch': int(args.rec_stop_epoch - 1),
            'ramp_start_epoch': int(args.rec_stop_epoch),
            'ramp_epochs': int(args.rec_rampdown_epochs),
            'relative_floor': float(args.rec_floor),
        },
        'hcl_margin': {
            'start_epoch': int(args.hcl_warmup_epoch),
            'end_before_epoch': int(args.adv_warmup_epoch),
            'start': float(args.hcl_margin_start),
            'end': float(args.hcl_margin_end),
        },
        'boundary': {
            'source_dbr_start_epoch': int(args.adv_warmup_epoch),
            'first_reference_calibration_epoch': int(args.first_calibrate_epoch),
            'novel_separation_start_epoch': int(args.novel_sep_start_epoch),
        },
        'data': {
            'hvg': int(args.hvg),
            'batch_size_maximum': int(args.batch_size),
            'source_label_ratio': float(args.source_label_ratio),
            'intra_standardization': 'joint_gene_wise_zscore',
            'cross_standardization': args.cross_standardization,
        },
        'target_neighborhood_consistency': {
            'weight': float(args.lambda_tnc),
            'start_epoch': int(args.tnc_start_epoch),
            'k': int(args.tnc_k),
            'margin': float(args.tnc_margin),
            'ontology_gate': str(args.tnc_ontology_gate),
            'ontology_gate_strength': float(args.tnc_ontology_gate_strength),
            'ontology_temperature': float(args.tnc_ontology_temperature),
            'confidence_gate_after_reference_alpha': bool(args.tnc_confidence_gate),
            'confidence_temperature': float(args.tnc_confidence_temperature),
            'positive_floor': float(args.tnc_positive_floor),
            'pre_alpha_scale': float(args.tnc_pre_alpha_scale),
            'safe_k_multiplier': int(args.tnc_safe_k_multiplier),
            'negative_gap': float(args.tnc_negative_gap),
            'triplet_weight': float(args.tnc_triplet_weight),
            'ontology_min_depth': int(args.tnc_ontology_min_depth),
            'uses_target_labels': False,
            'uses_target_class_cardinality': False,
            'gradient_target': 'target_z_only',
        },
        'evaluation': {
            'track1': 'GT-novel -> L2 -> fixed KMeans(K_novel) -> Hungarian',
            'track1_kmeans_seed': int(args.eval_kmeans_seed),
            'track1_kmeans_n_init': int(args.eval_kmeans_n_init),
            'track2': 'predicted-novel -> L2 -> fixed Leiden; no K',
            'track2_resolution': float(args.leiden_res),
            'track2_n_neighbors': int(args.leiden_n_neighbors),
            'track2_random_state': int(args.leiden_random_state),
        },
    }
    source_only_provenance = getattr(args, '_source_only_hvg_provenance', None)
    if source_only_provenance is not None:
        summary['source_only_hvg'] = source_only_provenance
    return summary


def _metrics_dict(values):
    auroc, overall_w, overall_j, known, novel, novel_macro = values
    return {
        'auroc': float(auroc),
        'overall_w': float(overall_w),
        'overall_j': float(overall_j),
        'known_acc': float(known),
        'novel_acc': float(novel),
        'novel_macro': float(novel_macro),
    }


# =============================================================================
# 6. 主训练循环
# =============================================================================

def train(args):

    _validate_protocol_configuration(args)

    # 0. 可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    probe_only = bool(
        args.protocol_profile == 'source_only_hvg'
        and args.source_only_episode_probe_only
    )
    device = torch.device(
        'cpu' if probe_only else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"[Setup] Device: {device}  Seed: {args.seed}")
    print(
        f"[Protocol] source_HCL={args.lambda_hcl}, "
        f"target_HCL={args.lambda_target_hcl}, "
        f"DBR={args.lambda_adv} "
    )
    # 1. Ontology
    ontology = OntologyManager(
        args.ontology_path,
        device                 = device,
        coarse_depth_threshold = args.coarse_depth_threshold,
    )
    # id2name 已由 OntologyManager.__init__ 自動建立
    print(f"[Ontology] id2name ready: {len(ontology.id2name)} entries")

    # 2. 数据
    episode_provenance = None
    if args.protocol_profile == 'source_only_hvg':
        (src_train_loader, src_eval_loader, tgt_eval_loader, tgt_train_loader,
         input_dim, unique_src_ids, novel_list, le,
         episode_provenance) = prepare_source_only_hvg_episode(args, ontology)
        args._source_only_hvg_provenance = episode_provenance
    elif getattr(args, 'cross_dataset', False):
        from cross_dataset_bridge import prepare_cross_dataset

        (src_train_loader, src_eval_loader, tgt_eval_loader, tgt_train_loader,
         input_dim, unique_src_ids, novel_list, le) = prepare_cross_dataset(
            args, ontology
        )
    else:
        (src_train_loader, src_eval_loader, tgt_eval_loader, tgt_train_loader,
         input_dim, unique_src_ids, novel_list, le) = prepare_datasets_robust(
            args, ontology
        )

    schedule_summary = _schedule_summary(args)
    if probe_only:
        probe_report = {
            'probe_only': True,
            'model_constructed': False,
            'optimizer_constructed': False,
            'checkpoint_created': False,
            'target_performance_evaluated': False,
            'source_loader_tensor_count': len(src_train_loader.dataset.tensors),
            'source_eval_loader_tensor_count': len(src_eval_loader.dataset.tensors),
            'target_training_loader_tensor_count': len(tgt_train_loader.dataset.tensors),
            'post_final_eval_loader_tensor_count': len(tgt_eval_loader.dataset.tensors),
            'input_dim': int(input_dim),
            'schedule': schedule_summary,
            'provenance': episode_provenance,
        }
        print("[source_only_hvg probe] " + json.dumps(probe_report, sort_keys=True))
        return probe_report

    # 3. 模型
    model = scOLAR(
        input_dim   = input_dim,
        num_classes = ontology.num_classes,
        use_zinb    = args.use_zinb,
    ).to(device)

    # [F4] buffer 注册：必须在 .to(device) 之后立即调用
    model.set_ontology_info(ontology.fine_to_coarse_mask, unique_src_ids)
    print(f"[Model] Buffers registered: "
          f"fine_to_coarse={tuple(ontology.fine_to_coarse_mask.shape)}, "
          f"src_ids={len(unique_src_ids)}")

    # 4. Criterion & Optimizer
    # hcl_weight=0.0：warm-up 阶段不启用 HCL
    criterion = scOLARLoss(
        ontology, 
        hcl_weight=0.0,
        rec_type = 'zinb' if args.use_zinb else 'mse',
        rec_weight=args.rec_weight,
        hcl_margin_start=args.hcl_margin_start,
        hcl_margin_end=args.hcl_margin_end,
        ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # LR 在每个 epoch 开始时由显式 cosine 公式设置；不读取任何 validation metric。
    # 不使用 ReduceLROnPlateau，避免 target 或其他运行期指标进入训练控制。
    scheduler = None

    # 5. 动态调度参数
    # 训练阶段划分（对应论文 warm-up 策略）：
    #   [0, HCL_WARMUP)          : L_CE + L_recon（表示稳健性）
    #   [HCL_WARMUP, ADV_WARMUP) : + L_HCL（原型几何约束）
    #   [ADV_WARMUP, end]        : + L_dec-adv（决策去偏）+ rec_weight 衰减
    HCL_WARMUP = args.hcl_warmup_epoch
    ADV_WARMUP = args.adv_warmup_epoch
    FIRST_CALIBRATE = args.first_calibrate_epoch

    # 6. HCL 动态 Margin 调度（论文创新点）[F7]
    # 线性从 hcl_margin_start 衰减到 hcl_margin_end
    def get_hcl_margin(epoch: int) -> float:
        if epoch < HCL_WARMUP:
            return args.hcl_margin_start
        progress = min(1.0, (epoch - HCL_WARMUP) / max(1, ADV_WARMUP - HCL_WARMUP))
        return args.hcl_margin_start + progress * (args.hcl_margin_end - args.hcl_margin_start)
   
    # [Modification 1] 建立 rec_weight 的闭式解
    def get_current_rec_weight(epoch: int) -> float:
        """
        三阶段 rec_weight 调度：
        [0,                adv_warmup)   : args.rec_weight
        [adv_warmup,       rec_stop)     : args.rec_weight * args.rec_decay_rate
        [rec_stop,         +inf)         : 线性衰减到 mid_weight * rec_floor
        """
        base = args.rec_weight
        mid_weight = base * args.rec_decay_rate

        if epoch < args.adv_warmup_epoch:
            return base

        if epoch < args.rec_stop_epoch:
            return mid_weight

        progress = min(
            1.0,
            (epoch - args.rec_stop_epoch) / max(1, args.rec_rampdown_epochs)
        )
        floor = float(getattr(args, 'rec_floor', 0.0))
        floor = min(max(floor, 0.0), 1.0)

        return mid_weight * (1.0 - progress * (1.0 - floor))

    def get_current_hcl_weight(epoch: int) -> float:
        """
        HCL 后期衰减调度：
        [0, HCL_WARMUP)              : 0.0
        [HCL_WARMUP, hcl_decay_start): args.lambda_hcl
        [hcl_decay_start, +inf)      : 线性衰减到 lambda_hcl * hcl_decay_floor
        """
        if epoch < HCL_WARMUP:
            return 0.0

        decay_start = getattr(args, 'hcl_decay_start', 999)
        if epoch < decay_start:
            return args.lambda_hcl

        decay_epochs = max(1, getattr(args, 'hcl_decay_epochs', 20))
        floor = min(max(getattr(args, 'hcl_decay_floor', 0.2), 0.0), 1.0)

        progress = min(1.0, (epoch - decay_start) / decay_epochs)
        return args.lambda_hcl * (1.0 - progress * (1.0 - floor))

    if args.protocol_profile != 'custom':
        # Executable correspondence check for Supplementary Table S4.
        schedule_checks = {
            'lambda_rec(epoch=1)': (get_current_rec_weight(1), 1.0),
            'lambda_rec(epoch=49)': (get_current_rec_weight(49), 1.0),
            'lambda_rec(epoch=50)': (get_current_rec_weight(50), 0.5),
            'lambda_rec(epoch=79)': (get_current_rec_weight(79), 0.5),
            'lambda_rec(epoch=80)': (get_current_rec_weight(80), 0.5),
            'lambda_rec(epoch=100)': (get_current_rec_weight(100), 0.0),
            'hcl_margin(epoch=20)': (get_hcl_margin(20), 0.3),
            'hcl_margin(epoch=50)': (get_hcl_margin(50), 0.05),
        }
        failed = [
            f"{name}: got {actual:.12g}, expected {expected:.12g}"
            for name, (actual, expected) in schedule_checks.items()
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        ]
        if failed:
            raise RuntimeError(
                "Executable Table S4 schedule check failed:\n  - "
                + "\n  - ".join(failed)
            )
        print(
            "[Protocol] Table S4 schedule check passed: "
            "lambda_rec 1.0 (ep1-49) -> 0.5 (ep50-79) -> 0 by ep100; "
            "HCL margin 0.3 at ep20 -> 0.05 at ep50."
        )

    # 7. Loss Meters
    meter_keys = ['total', 'proto', 'rec_src', 'rec_tgt',
                  'hcl_src', 'hcl_tgt', 'tnc', 'tnc_mutual', 'tnc_active',
                  'tnc_safe_neg', 'tnc_pair_weight', 'tnc_qnovel',
                  'tnc_pos_onto', 'tnc_neg_onto',
                  'adv_src', 'adv_tgt', 'novel_sep']
    meters = {k: AverageMeter(k) for k in meter_keys}

    alpha = None
    start_time = time.time()
    final_ckpt_file = os.path.join(args.output_dir, args.checkpoint_path)

    if args.calibrate_every <= 0:
        raise ValueError('--calibrate_every must be a positive integer')
    if args.lambda_novel_sep > 0 and args.first_calibrate_epoch > args.epochs:
        raise ValueError(
            'Novel separation is enabled but alpha is never calibrated within the training budget'
        )

    snapshot_epochs = _default_snapshot_epochs(args)
    calibration_plan = build_calibration_plan(
        src_eval_loader,
        base_seed=args.seed,
        num_mask=args.calibration_num_mask,
        n_simulations=args.calibration_simulations,
    )
    calibration_plan_path = str(
        Path(final_ckpt_file).with_suffix('.calibration_draws.json')
    )
    Path(calibration_plan_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calibration_plan_path).write_text(
        json.dumps(calibration_plan, indent=2), encoding='utf-8'
    )

    # Repository-level guard: target training path must expose exactly three tensors.
    assert isinstance(tgt_train_loader.dataset, TensorDataset)
    assert len(tgt_train_loader.dataset.tensors) == 3, (
        "Target training loader contains benchmark metadata; expected only "
        "(x, raw_x, size_factor)."
    )

    print(f"\n=== Training scOLAR | "
          f"Epochs={args.epochs} | HCL@ep{HCL_WARMUP} | Adv@ep{ADV_WARMUP} | "
          f"fixed-final checkpoint=epoch {args.epochs} ===")
    print("[Protocol] Target labels are unavailable to the training loader and are not used "
          "for LR scheduling, early stopping, or checkpoint selection.")
    print(
        f"[LR] Prespecified shared-horizon cosine: base={args.lr:.6g}, "
        f"min={args.min_lr:.6g}, horizon={args.lr_schedule_epochs}, "
        f"run_epochs={args.epochs}"
    )
    print(f"[Protocol] {json.dumps(schedule_summary, sort_keys=True)}")
    print(f"[Calibration] Frozen draws written to: {calibration_plan_path}")
    print(f"[Checkpoint] Prespecified snapshots: {sorted(snapshot_epochs)} + final {args.epochs}\n")

    # 8. 主循环
    for epoch in range(1, args.epochs + 1):
        current_lr = _cosine_learning_rate(
            epoch=epoch,
            schedule_epochs=args.lr_schedule_epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
        )
        _set_optimizer_lr(optimizer, current_lr)

        # Reference-only calibration is performed before the epoch that first uses alpha,
        # so epoch FIRST_CALIBRATE contains the complete boundary objective.
        should_calibrate = (
            epoch == FIRST_CALIBRATE
            or (epoch > FIRST_CALIBRATE and
                (epoch - FIRST_CALIBRATE) % args.calibrate_every == 0)
        )
        if should_calibrate:
            alpha = calibrate_threshold(
                model, src_eval_loader, device,
                epsilon=args.calibrate_epsilon,
                calibration_plan=calibration_plan,
            )
            if epoch == FIRST_CALIBRATE:
                print(
                    f"[Schedule] Epoch {epoch}: first reference calibration complete "
                    f"(alpha={alpha:.4f})"
                )
                if args.lambda_novel_sep > 0 and epoch >= args.novel_sep_start_epoch:
                    print(f"[Schedule] Epoch {epoch}: novel-separation activated")
                if args.lambda_target_hcl > 0:
                    print(
                        f"[Schedule] Epoch {epoch}: label-free target HCL activated "
                        f"with likely-known rule MLS >= alpha"
                    )

        model.train()
        for m in meters.values():
            m.reset()

        # --- 动态调度 ---
        target_hcl_weight = get_current_hcl_weight(epoch)
        if abs(criterion.hcl_weight - target_hcl_weight) > 1e-8:
            criterion.hcl_weight = target_hcl_weight
            if epoch >= HCL_WARMUP:
                print(f"[Schedule] Epoch {epoch}: hcl_weight -> {criterion.hcl_weight:.4f}")

        target_rec_weight = get_current_rec_weight(epoch)
        if abs(criterion.rec_weight - target_rec_weight) > 1e-8:
            criterion.rec_weight = target_rec_weight
            print(f"[Schedule] Epoch {epoch}: rec_weight -> {criterion.rec_weight:.4f}")

        if ADV_WARMUP <= epoch < args.adv_stop_epoch:
            lambda_adv = args.lambda_adv
        else:
            lambda_adv = 0.0
        if epoch == ADV_WARMUP and args.lambda_adv > 0:
            if args.adv_mode == 'source':
                print(f"[Schedule] Epoch {epoch}: source DBR activated (alpha-independent)")
            elif args.adv_mode == 'target':
                print(f"[Schedule] Epoch {epoch}: target DBR scheduled; waits for alpha")
        current_margin = get_hcl_margin(epoch)   # [F7]
        if epoch == args.tnc_start_epoch and args.lambda_tnc > 0:
            print(
                f"[Schedule] Epoch {epoch}: TNC activated "
                f"(lambda={args.lambda_tnc:.4f}, k={args.tnc_k}, "
                f"margin={args.tnc_margin:.3f}, pos_floor={args.tnc_positive_floor:.2f}, "
                f"safe_k={args.tnc_safe_k_multiplier}x, ontology_gate={args.tnc_ontology_gate})"
            )

        tgt_iter = cycle(tgt_train_loader)

        for batch_idx, (sx, sraw, ssf, sy, svalid) in enumerate(src_train_loader):
            sx, sraw, ssf = sx.to(device), sraw.to(device), ssf.to(device)
            sy, svalid    = sy.to(device), svalid.to(device)

            # Target training loader only exposes unlabelled inputs.
            tx, traw, tsf = next(tgt_iter)
            tx, traw, tsf = tx.to(device), traw.to(device), tsf.to(device)

            optimizer.zero_grad()

            # --- Forward ---
            s_out = model(sx)   # s_coarse_max 出现当且仅当 buffer 已注册
            t_out = model(tx)

            if not (_finite(s_out['logits_fine']) and _finite(s_out['z']) and
                    _finite(t_out['logits_fine']) and _finite(t_out['z'])):
                print(f"[NaN] non-finite forward at epoch {epoch}, skip batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            # === Loss A：L_CE + L_HCL_src（源域主分类 + 原型几何）===
            # hierarchy_triplet_aug 接收动态 margin [F7]
            src_triplets = torch.empty((0, 3), dtype=torch.long, device=device)
            if criterion.hcl_weight > 0:
                src_triplets = hierarchy_triplet_aug(
                    sy, ontology, device=device, margin=current_margin
                )

            loss, l_dict = criterion(
                s_out, sy, sraw, src_triplets,
                sf=ssf, mask=svalid,
                epoch=epoch, total_epochs=args.epochs,
                margin_override=current_margin,
            )
            meters['proto'].update(l_dict.get('proto', 0.0))
            meters['hcl_src'].update(l_dict.get('hcl', 0.0))
            meters['rec_src'].update(l_dict.get('rec', 0.0))

            # === Loss B：L_recon_tgt（目标域无监督重构，DA 稳健性）===
            # [F8] rec_tgt meter 正确记录
            if criterion.rec_weight > 0:
                if args.use_zinb:
                    l_rec_tgt = criterion.zinb(
                        traw, t_out['recon']['mean'],
                        t_out['recon']['disp'], t_out['recon']['drop'], sf=tsf,
                    )
                else:
                    l_rec_tgt = F.mse_loss(t_out['recon'], traw)
                loss += criterion.rec_weight * l_rec_tgt
                meters['rec_tgt'].update(l_rec_tgt.item())

            # === Loss C：L_TNC-v2（保守 target-z 结构保持；不使用 target GT / K_novel）===
            if args.lambda_tnc > 0 and epoch >= args.tnc_start_epoch:
                l_tnc, tnc_stats = compute_target_neighborhood_consistency_loss(
                    z_tgt=t_out['z'],
                    logits_fine=t_out['logits_fine'],
                    known_ids=unique_src_ids.to(device),
                    ancestor_matrix=ontology.ancestor_matrix.to(device),
                    depths=ontology.depths.to(device),
                    alpha=alpha,
                    epoch=epoch,
                    k=args.tnc_k,
                    margin=args.tnc_margin,
                    positive_floor=args.tnc_positive_floor,
                    pre_alpha_scale=args.tnc_pre_alpha_scale,
                    safe_k_multiplier=args.tnc_safe_k_multiplier,
                    negative_gap=args.tnc_negative_gap,
                    triplet_weight=args.tnc_triplet_weight,
                    ontology_gate=args.tnc_ontology_gate,
                    ontology_gate_strength=args.tnc_ontology_gate_strength,
                    ontology_temperature=args.tnc_ontology_temperature,
                    ontology_min_depth=args.tnc_ontology_min_depth,
                    confidence_gate=args.tnc_confidence_gate,
                    confidence_temperature=args.tnc_confidence_temperature,
                )
                loss += args.lambda_tnc * l_tnc
                meters['tnc'].update(l_tnc.item())
                meters['tnc_mutual'].update(tnc_stats['mutual_frac'])
                meters['tnc_active'].update(tnc_stats['active_frac'])
                meters['tnc_safe_neg'].update(tnc_stats['safe_neg_frac'])
                meters['tnc_pair_weight'].update(tnc_stats['pair_weight'])
                meters['tnc_qnovel'].update(tnc_stats['q_novel'])
                meters['tnc_pos_onto'].update(tnc_stats['pos_onto'])
                meters['tnc_neg_onto'].update(tnc_stats['neg_onto'])
            
            # === Loss C2：legacy target-HCL（辅助配置；TNC-v2 必须保持关闭）===
            # 旧代码使用由真实 novel partition 派生的 tvalid 排除 novel cells，
            # 这会让 target ground truth 进入辅助配置的梯度路径。现在只在
            # reference-calibrated alpha 可用后，对模型判为 likely-known 的 target cells
            # 构造 pseudo-label HCL，并把 pseudo label 限制到 observed source prototypes。
            if args.lambda_target_hcl > 0 and alpha is not None:
                with torch.no_grad():
                    target_mls = t_out['logits_fine'].max(dim=1).values
                    likely_known = target_mls >= alpha
                    src_ids_dev = unique_src_ids.to(device)
                    if likely_known.any():
                        known_logits = t_out['logits_fine'][likely_known][:, src_ids_dev]
                        pseudo_local = known_logits.argmax(dim=1)
                        tgt_pseudo_y = src_ids_dev[pseudo_local]
                    else:
                        tgt_pseudo_y = None

                if tgt_pseudo_y is not None and tgt_pseudo_y.numel() > 0:
                    tgt_triplets = hierarchy_triplet_aug(
                        tgt_pseudo_y,
                        ontology,
                        device=device,
                        margin=current_margin,
                    )

                    if tgt_triplets.size(0) > 0:
                        l_hcl_tgt = criterion.hcl(
                            F.normalize(t_out['prototypes'], p=2, dim=1),
                            tgt_triplets,
                            margin=current_margin,
                        )

                        loss += args.lambda_target_hcl * l_hcl_tgt
                        meters['hcl_tgt'].update(l_hcl_tgt.item())

            # === Loss D：L_dec-adv ===
            # Source DBR 只依赖 reference labels 和 coarse/fine geometry，不依赖 alpha。
            if lambda_adv > 0 and args.adv_mode == 'source' and svalid.any():
                l_adv_src = compute_adv_loss(
                    s_coarse_max=s_out['s_coarse_max'][svalid],
                    logits_fine=s_out['logits_fine'][svalid],
                    targets=sy[svalid],
                    scale=model.scale.item(),
                    beta=args.beta,
                    adv_margin=args.adv_margin,
                    debug=(args.debug and epoch % 10 == 0),
                    global_step=epoch * len(src_train_loader) + batch_idx,
                    log_every=100,
                )
                loss += lambda_adv * l_adv_src
                meters['adv_src'].update(l_adv_src.item())

            # Target-side DBR 是负对照/辅助变体，因 likely-known gating 才需要 alpha。
            elif (lambda_adv > 0 and args.adv_mode == 'target' and alpha is not None
                  and 's_coarse_max' in t_out):
                l_adv_tgt = compute_target_adv_loss(
                    z_tgt_norm=F.normalize(t_out['z'], p=2, dim=1),
                    s_coarse_max_tgt=t_out['s_coarse_max'],
                    logits_fine_tgt=t_out['logits_fine'],
                    p_norm=F.normalize(t_out['prototypes'], p=2, dim=1),
                    alpha=alpha,
                    beta=args.beta,
                    adv_margin=args.adv_margin,
                    temp=getattr(args, 'adv_novel_temp', 5.0),
                    debug=(args.debug and epoch % 10 == 0),
                    global_step=epoch * len(src_train_loader) + batch_idx,
                    log_every=100,
                )
                loss += lambda_adv * l_adv_tgt
                meters['adv_tgt'].update(l_adv_tgt.item())
            
            # === Loss E：L_novel-sep（target likely-novel cells 远离 known prototypes）===
            if (
                getattr(args, 'lambda_novel_sep', 0.0) > 0
                and epoch >= getattr(args, 'novel_sep_start_epoch', 60)
                and alpha is not None
            ):
                l_novel_sep = compute_novel_sep_loss(
                    z_tgt=t_out['z'],
                    logits_fine=t_out['logits_fine'],
                    prototypes=t_out['prototypes'],
                    known_ids=unique_src_ids.to(device),
                    alpha=alpha,
                    temp=args.novel_sep_temp,
                    margin=args.novel_sep_margin,
                )
                loss += args.lambda_novel_sep * l_novel_sep
                meters['novel_sep'].update(l_novel_sep.item())

            # --- Backward ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            meters['total'].update(loss.item())

        # --- Epoch 日志 ---
        if epoch % 5 == 0:
            elapsed = (time.time() - start_time) / 60
            print(
                f"Ep {epoch:03d}/{args.epochs} [{elapsed:.1f}min] | "
                f"lr={current_lr:.6g} | "
                f"total={meters['total'].avg:.4f} | "
                f"proto={meters['proto'].avg:.4f} | "
                f"rec={meters['rec_src'].avg:.4f}/{meters['rec_tgt'].avg:.4f} "
                f"(w={criterion.rec_weight:.4f}) | "
                f"hcl={meters['hcl_src'].avg:.4f}/{meters['hcl_tgt'].avg:.4f} "
                f"(w={criterion.hcl_weight:.4f}) | "
                f"tnc={meters['tnc'].avg:.4f} "
                f"(w={args.lambda_tnc:.4f}, mutual={meters['tnc_mutual'].avg:.3f}, "
                f"active={meters['tnc_active'].avg:.3f}, safeNeg={meters['tnc_safe_neg'].avg:.3f}, "
                f"pairW={meters['tnc_pair_weight'].avg:.3f}, qNov={meters['tnc_qnovel'].avg:.3f}, "
                f"onto+/-={meters['tnc_pos_onto'].avg:.3f}/{meters['tnc_neg_onto'].avg:.3f}) | "
                f"adv_src={meters['adv_src'].avg:.4f} | adv_tgt={meters['adv_tgt'].avg:.4f} | "
                f"m={current_margin:.3f}"
            )

        # 按预先声明的 epoch 无条件保存 snapshots；不读取 target metrics。
        if epoch in snapshot_epochs and epoch != args.epochs:
            snapshot_path = _snapshot_checkpoint_path(final_ckpt_file, epoch)
            _save_checkpoint(
                snapshot_path, model, optimizer, scheduler, epoch, args,
                role='prespecified_snapshot',
                selection_rule=f'fixed_epoch_{epoch}',
                calibration_plan=calibration_plan,
                calibration_plan_path=calibration_plan_path,
            )

    # 9. 固定最终 checkpoint：必须先保存，再访问任何 target benchmark label。
    _save_checkpoint(
        final_ckpt_file, model, optimizer, scheduler, args.epochs, args,
        role='primary_final',
        selection_rule=f'fixed_final_epoch_{args.epochs}',
        calibration_plan=calibration_plan,
        calibration_plan_path=calibration_plan_path,
    )

    print("\n=== Primary Post-training Evaluation (fixed final checkpoint) ===")
    load_checkpoint(final_ckpt_file, model, device=device)
    alpha = calibrate_threshold(
        model, src_eval_loader, device,
        epsilon=args.calibrate_epsilon,
        calibration_plan=calibration_plan,
    )
    final_eval_details = {}
    final_values = validate_osr(
        model, tgt_eval_loader, alpha, device,
        ontology, unique_src_ids, epoch="FINAL",
        le=le,
        novel_list=novel_list,
        leiden_res=args.leiden_res,
        lcc_temperature=args.lcc_temperature,
        eval_kmeans_seed=args.eval_kmeans_seed,
        eval_kmeans_n_init=args.eval_kmeans_n_init,
        leiden_n_neighbors=args.leiden_n_neighbors,
        leiden_random_state=args.leiden_random_state,
        run_deployment_eval=(args.protocol_profile != 'source_only_hvg'),
        run_lcc_eval=(args.protocol_profile != 'source_only_hvg'),
        eval_details=final_eval_details,
    )
    final_metrics = _metrics_dict(final_values)
    final_checkpoint_sha256 = _sha256_file(Path(final_ckpt_file))
    dual_eval_path = Path(final_ckpt_file).with_suffix('.dual_evaluation.json')
    dual_eval_payload = {
        'checkpoint': str(final_ckpt_file),
        'checkpoint_sha256': final_checkpoint_sha256,
        'checkpoint_epoch': int(args.epochs),
        'selection_rule': f'fixed_final_epoch_{args.epochs}',
        'target_metrics_used_for_checkpoint_selection': False,
        'target_training_labels_present': False,
        'benchmark_target_cells_present': (
            False if args.protocol_profile == 'source_only_hvg' else None
        ),
        'source_only_hvg_provenance': episode_provenance,
        'calibration_plan_path': calibration_plan_path,
        'calibration_draw_seed': calibration_plan['draw_seed'],
        'schedule': schedule_summary,
        'primary_metrics': final_metrics,
        **final_eval_details,
    }
    dual_eval_path.write_text(json.dumps(dual_eval_payload, indent=2), encoding='utf-8')
    print(f"[Evaluation] Wrote dual-track report: {dual_eval_path}")

    if args.protocol_profile == 'source_only_hvg':
        track1 = final_eval_details.get('track1_oracle_kmeans')
        if track1 is None:
            raise RuntimeError(
                'source_only_hvg fixed-final scoring did not produce Track-1 metrics'
            )
        source_only_result = {
            'dataset': args.dataset,
            'dev_seed': int(args.seed),
            'fold': int(args.source_only_fold),
            'hvg': int(args.hvg),
            'novel_weighted': float(final_metrics['novel_acc']),
            'novel_macro': float(final_metrics['novel_macro']),
            'ari': float(track1['ari']),
            'known': float(final_metrics['known_acc']),
            'overall_j': float(final_metrics['overall_j']),
            'checkpoint': str(final_ckpt_file),
            'checkpoint_sha256': final_checkpoint_sha256,
            'checkpoint_epoch': int(args.epochs),
            'checkpoint_selection': f'fixed_final_epoch_{args.epochs}',
            'target_gt_used_for_selection': False,
            'target_training_labels_present': False,
            'benchmark_target_cells_present': False,
            'source_only_hvg_provenance': episode_provenance,
        }
        source_only_result_path = Path(final_ckpt_file).with_suffix(
            '.source_only_hvg_result.json'
        )
        source_only_result_path.write_text(
            json.dumps(source_only_result, indent=2), encoding='utf-8'
        )
        print(f"[source_only_hvg] Wrote fixed-final result: {source_only_result_path}")

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*64}")
    print(f"  Primary checkpoint       : fixed epoch {args.epochs}")
    print(f"  Target used in selection : NO")
    print(f"  Final Overall (W)        : {final_metrics['overall_w']*100:.1f}%")
    print(f"  Final Overall (J)        : {final_metrics['overall_j']*100:.1f}%")
    print(f"  Final AUROC              : {final_metrics['auroc']:.4f}")
    print(f"  Final Known Acc          : {final_metrics['known_acc']*100:.1f}%")
    print(f"  Final Novel Acc          : {final_metrics['novel_acc']*100:.1f}%")
    print(f"  Final Novel Macro        : {final_metrics['novel_macro']*100:.1f}%")
    print(f"  Total Time               : {total_time:.1f} min")
    print(f"{'='*64}\n")

    # 可选：训练全部结束后，离线评估固定 snapshots，量化旧 target-selected
    # protocol 的 optimistic gap。该诊断绝不回写 optimizer、scheduler 或 checkpoint。
    if args.posthoc_oracle_diagnostic:
        diagnostic = {
            'protocol': {
                'primary_selection': f'fixed_final_epoch_{args.epochs}',
                'target_metrics_used_during_training': False,
                'oracle_diagnostic_is_posthoc_only': True,
                'calibration_draw_seed': calibration_plan['draw_seed'],
                'calibration_plan_path': calibration_plan_path,
                'heldout_class_ids': calibration_plan['heldout_class_ids'],
            },
            'checkpoints': {},
        }

        candidate_epochs = sorted(snapshot_epochs | {args.epochs})
        for snapshot_epoch in candidate_epochs:
            path = (final_ckpt_file if snapshot_epoch == args.epochs
                    else _snapshot_checkpoint_path(final_ckpt_file, snapshot_epoch))
            if not os.path.exists(path):
                print(f"[Posthoc] Skip missing snapshot: {path}")
                continue

            load_checkpoint(path, model, device=device)
            snapshot_alpha = calibrate_threshold(
                model, src_eval_loader, device,
                epsilon=args.calibrate_epsilon,
                calibration_plan=calibration_plan,
            )
            values = validate_osr(
                model, tgt_eval_loader, snapshot_alpha, device,
                ontology, unique_src_ids, epoch=f"POSTHOC_E{snapshot_epoch}",
                le=le,
                novel_list=novel_list,
                leiden_res=args.leiden_res,
                lcc_temperature=args.lcc_temperature,
                eval_kmeans_seed=args.eval_kmeans_seed,
                eval_kmeans_n_init=args.eval_kmeans_n_init,
                leiden_n_neighbors=args.leiden_n_neighbors,
                leiden_random_state=args.leiden_random_state,
                run_deployment_eval=False,
            )
            diagnostic['checkpoints'][str(snapshot_epoch)] = _metrics_dict(values)

        if diagnostic['checkpoints']:
            oracle_epoch, oracle_metrics = max(
                diagnostic['checkpoints'].items(),
                key=lambda item: item[1]['overall_j'],
            )
            fixed = diagnostic['checkpoints'].get(str(args.epochs), final_metrics)
            diagnostic['oracle_target_selected'] = {
                'epoch': int(oracle_epoch),
                **oracle_metrics,
                'optimism_gap_over_fixed_final_pp':
                    100.0 * (oracle_metrics['overall_j'] - fixed['overall_j']),
            }
            print(
                f"[Posthoc oracle diagnostic] best target-Overall epoch={oracle_epoch}; "
                f"gap over fixed final="
                f"{diagnostic['oracle_target_selected']['optimism_gap_over_fixed_final_pp']:+.2f} pp"
            )

        diagnostic_path = Path(final_ckpt_file).with_suffix('.checkpoint_diagnostic.json')
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
        print(f"[Posthoc] Wrote diagnostic: {diagnostic_path}")

        # Restore primary model state before returning.
        load_checkpoint(final_ckpt_file, model, device=device)

    return final_metrics


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scOLAR Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =========================
    # 基础配置
    # =========================
    g_base = parser.add_argument_group("Base")
    g_base.add_argument('--seed', type=int, default=42)
    g_base.add_argument('--dataset', type=str, default='Quake_10x')
    g_base.add_argument('--ontology_path', type=str, default='data/cl.pt')
    g_base.add_argument('--checkpoint_path', type=str, default='final_model.pth',
                        help='Path for the prespecified fixed-final checkpoint')
    g_base.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save models and logs')
    g_base.add_argument(
        '--protocol_profile',
        choices=[
            'paper_primary', 'paper_schedule', 'source_only_hvg',
            'dev_tnc', 'dev_tnc_v2', 'dev_tnc_v2_zero', 'custom',
        ],
        default='paper_primary',
        help='Validation level for manuscript-aligned defaults. Use paper_schedule '
             'for prespecified robustness sweeps, source_only_hvg for the frozen '
             'source-derived 2k/4k screen, and custom for documented deviations.'
    )
    g_base.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable verbose tensor/DBR diagnostics')
    # =========================
    # 数据与划分
    # =========================
    g_data = parser.add_argument_group("Data")
    g_data.add_argument('--hvg', type=int, default=2000,
                        help='Number of highly variable genes')
    g_data.add_argument('--batch_size', type=int, default=1024,
                        help='Maximum mini-batch size; final partial batches are retained')
    g_data.add_argument('--ratio_shared', type=float, default=0.6,
                        help='Cs fraction (used for ratio-based split only)')
    g_data.add_argument('--ratio_src_priv', type=float, default=0.2,
                        help='Cr fraction (used for ratio-based split only)')
    g_data.add_argument('--source_label_ratio', type=float, default=0.5,
                        help='For intra-dataset benchmarks, probability that a shared cell is assigned '
                             'exclusively to source; the complement goes to target. Source-private cells '
                             'remain 100%% in source, matching the released scBOL split protocol.')
    g_data.add_argument('--src_min_per_class', type=int, default=10,
                        help='Deprecated compatibility argument; not used by the benchmark-exclusive split')
    g_data.add_argument('--n_cs_override', type=int, default=None,
                        help='Override n_cs from registry (Table S8: set n_cr=0 simultaneously)')
    g_data.add_argument('--n_cr_override', type=int, default=None,
                        help='Override n_cr from registry (Table S8: always 0)')  
    g_data.add_argument('--novel_ratio', type=float, default=1.0,
                    help='Fraction of Ct (novel) cells included in target '
                         '(Table S11 robustness; 1.0=default=all Ct cells)')
    g_data.add_argument(
        '--source_only_manifest', type=str, default=None,
        help='Frozen source_only_hvg_partition_v1 manifest; required only by source_only_hvg'
    )
    g_data.add_argument(
        '--source_only_fold', type=int, choices=list(SOURCE_ONLY_HVG_ALLOWED_FOLDS),
        default=None, help='Frozen pseudo-novel class fold for source_only_hvg'
    )
    g_data.add_argument(
        '--source_only_episode_probe_only', action='store_true',
        help='Validate/reconstruct/preprocess the frozen episode and exit before model creation'
    )

    # =========================
    # 模型配置
    # =========================
    g_model = parser.add_argument_group("Model")
    g_model.add_argument('--use_zinb', action=argparse.BooleanOptionalAction, default=True,
                         help='Use ZINB reconstruction loss')
    g_model.add_argument('--coarse_depth_threshold', type=int, default=3,
                         help='Depth threshold for coarse nodes in OntologyManager')

    # =========================
    # 训练主超参
    # =========================
    g_train = parser.add_argument_group("Training")
    g_train.add_argument('--epochs', type=int, default=100,
                         help='Prespecified full training budget; final epoch is primary checkpoint')
    g_train.add_argument('--lr', type=float, default=1e-3,
                         help='Base learning rate of the prespecified cosine schedule')
    g_train.add_argument('--min_lr', type=float, default=1e-5,
                         help='Minimum learning rate of the prespecified cosine schedule')
    g_train.add_argument('--lr_schedule_epochs', type=int, default=100,
                         help='Shared absolute cosine horizon used by every run; '
                              'cross-dataset 60-epoch runs use its first 60 steps')
    g_train.add_argument('--snapshot_epochs', type=int, nargs='*', default=None,
                         help='Optional prespecified snapshots. Default: calibration-grid '
                              'epochs for intra-dataset; final-20/final-10 for cross-dataset')
    g_train.add_argument('--posthoc_oracle_diagnostic',
                         action=argparse.BooleanOptionalAction, default=False,
                         help='After training only, score snapshots on target GT to quantify optimism')

    # =========================
    # 损失权重与调度
    # =========================
    g_sched = parser.add_argument_group("Loss Schedule")
    g_sched.add_argument(
        '--lambda_hcl',
        type=float,
        default=0.1,
        help='Weight for source HCL'
    )

    g_sched.add_argument(
        '--lambda_target_hcl',
        type=float,
        default=0.0,
        help='Legacy prototype-level target pseudo-label HCL; TNC development keeps this at 0'
    )

    g_sched.add_argument('--lambda_tnc', type=float, default=0.0,
                         help='Weight for target-neighborhood consistency; 0 reproduces strict baseline')
    g_sched.add_argument('--tnc_start_epoch', type=int, default=20,
                         help='Epoch to activate weak positive-only TNC; v2 aligns this with HCL warm-up')
    g_sched.add_argument('--tnc_k', type=int, default=5,
                         help='Within-batch k for reliable mutual-kNN positive candidate mining')
    g_sched.add_argument('--tnc_margin', type=float, default=0.05,
                         help='Fixed cosine margin for conservative post-alpha safe triplets')
    g_sched.add_argument('--tnc_ontology_gate', choices=['off', 'soft'], default='soft',
                         help='Ontology conditioning of reliable positive pairs; never defines negative pairs')
    g_sched.add_argument('--tnc_ontology_gate_strength', type=float, default=0.25,
                         help='Soft deep-ancestor ontology contribution in [0,1] for positive ranking/weighting')
    g_sched.add_argument('--tnc_ontology_temperature', type=float, default=2.0,
                         help='Temperature for observed-prototype probabilities used in deep-ancestor compatibility')
    g_sched.add_argument('--tnc_confidence_gate',
                         action=argparse.BooleanOptionalAction, default=True,
                         help='After reference alpha exists, downweight likely-known/likely-novel cross-boundary positives')
    g_sched.add_argument('--tnc_confidence_temperature', type=float, default=5.0,
                         help='Temperature of the reference-alpha confidence compatibility gate')
    g_sched.add_argument('--tnc_positive_floor', type=float, default=0.80,
                         help='Stop positive pull once mutual-neighbour cosine reaches this floor')
    g_sched.add_argument('--tnc_pre_alpha_scale', type=float, default=0.10,
                         help='Relative weight of weak positive-only TNC before reference alpha exists')
    g_sched.add_argument('--tnc_safe_k_multiplier', type=int, default=2,
                         help='Exclude the closest multiplier*k neighbours from negative mining')
    g_sched.add_argument('--tnc_negative_gap', type=float, default=0.02,
                         help='Negative candidate must already be at least this much farther than positive')
    g_sched.add_argument('--tnc_triplet_weight', type=float, default=0.25,
                         help='Relative weight of conservative post-alpha triplet term')
    g_sched.add_argument('--tnc_ontology_min_depth', type=int, default=2,
                         help='Minimum ontology depth retained when computing ancestor-Jaccard compatibility')

    g_sched.add_argument(
        '--lambda_adv',
        type=float,
        default=0.05,
        help='Weight for L_dec-adv'
    )

    g_sched.add_argument('--lambda_novel_sep', type=float, default=0.01,
                     help='Weight for novel separation loss')
    g_sched.add_argument('--novel_sep_temp', type=float, default=5.0,
                        help='Temperature for novel separation weighting')
    g_sched.add_argument('--novel_sep_margin', type=float, default=0.1,
                        help='Margin for novel separation from known prototypes')
    g_sched.add_argument('--novel_sep_start_epoch', type=int, default=60,
                        help='Epoch to start novel separation loss (effective only '
                            'after alpha calibration at FIRST_CALIBRATE)')

    g_sched.add_argument('--rec_weight', type=float, default=1.0,
                         help='Base reconstruction weight')
    g_sched.add_argument('--rec_decay_rate', type=float, default=0.5,
                         help='Mid-stage decay rate for rec_weight after ADV_WARMUP')
    g_sched.add_argument('--rec_stop_epoch', type=int, default=80,
                         help='Epoch after which rec_weight starts ramping down')
    g_sched.add_argument('--rec_rampdown_epochs', type=int, default=20,
                         help='Epochs over which rec ramps from mid to floor')
    g_sched.add_argument('--rec_floor', type=float, default=0.0,
                         help='Relative floor of rec_weight after rampdown (0=off)')

    g_sched.add_argument('--hcl_warmup_epoch', type=int, default=20,
                         help='Epoch to activate HCL')
    g_sched.add_argument('--adv_warmup_epoch', type=int, default=50,
                         help='Epoch to activate L_dec-adv')

    g_sched.add_argument('--hcl_decay_start', type=int, default=999,
                         help='Epoch to start HCL weight decay (default=off)')
    g_sched.add_argument('--hcl_decay_epochs', type=int, default=20,
                         help='Epochs to ramp HCL from full to floor')
    g_sched.add_argument('--hcl_decay_floor', type=float, default=0.2,
                         help='Minimum fraction of lambda_hcl to retain')

    g_sched.add_argument('--hcl_margin_start', type=float, default=0.3,
                         help='Initial large margin for HCL triplet (early training)')
    g_sched.add_argument('--hcl_margin_end', type=float, default=0.05,
                         help='Final small margin for HCL triplet (late training)')

    # =========================
    # 对抗分支
    # =========================
    g_adv = parser.add_argument_group("Adversarial")
    g_adv.add_argument('--beta', type=float, default=0.3,
                       help='Minimum required fine-coarse cosine gap in L_dec-adv '
                            '(gap = fine_cos - S_coarse; penalise when gap < beta)')
    g_adv.add_argument('--adv_margin', type=float, default=0.05,
                       help='Slack in ReLU((beta - adv_margin) - gap)')
    g_adv.add_argument('--adv_stop_epoch', type=int, default=999,
                       help='Epoch to stop L_dec-adv')
    g_adv.add_argument('--adv_mode', type=str, default='source',
                   choices=['source', 'target', 'off'],
                   help='Adversarial loss application mode')                   

    # =========================
    # 推理、标定、解释
    # =========================
    g_eval = parser.add_argument_group("Calibration & LCC")
    g_eval.add_argument('--calibrate_epsilon', type=float, default=0.05,
                        help='Known-class TPR lower bound = 1 - epsilon')
    g_eval.add_argument('--calibration_num_mask', type=int, default=10,
                        help='Number of reference classes held out in each frozen calibration draw')
    g_eval.add_argument('--calibration_simulations', type=int, default=20,
                        help='Number of frozen held-out-class draws per run')
    g_eval.add_argument('--first_calibrate_epoch', type=int, default=60,
                        help='First epoch whose training updates use reference-calibrated alpha')
    g_eval.add_argument('--calibrate_every', type=int, default=10,
                        help='Refresh alpha every N epochs using reference data only')
    g_eval.add_argument('--lcc_temperature', type=float, default=2.0,
                        help='Softmax temperature for LCC coarse projection (>1 = softer)')
    g_eval.add_argument('--leiden_res', type=float, default=1.0,
                        help='Fixed Leiden resolution for Track-2 deployment clustering and LCC')
    g_eval.add_argument('--leiden_n_neighbors', type=int, default=15,
                        help='Fixed kNN size for the common Track-2 Leiden evaluator')
    g_eval.add_argument('--leiden_random_state', type=int, default=0,
                        help='Frozen random seed for the common Track-2 Leiden evaluator')
    g_eval.add_argument('--eval_kmeans_seed', type=int, default=0,
                        help='Frozen evaluator seed for Track-1 oracle-K KMeans; independent of training seed')
    g_eval.add_argument('--eval_kmeans_n_init', type=int, default=10,
                        help='KMeans restarts selected by inertia only; target GT never selects a restart')

    # =========================
    # 额外的 Cross-Dataset Probe（Frame-A）
    # =========================
    g_cross = parser.add_argument_group('cross-dataset probe')
    g_cross.add_argument('--cross_dataset', action='store_true',
                        help='run Frame-A cross-dataset open-set probe')
    g_cross.add_argument('--cross_h5ad', type=str,
                        default='data/ALIGNED_Mus_musculus_Mammary_Gland.h5ad')
    g_cross.add_argument('--cross_ref_name', type=str,
                        default='Quake_Smart-seq2_Mammary_Gland')
    g_cross.add_argument('--cross_tgt_name', type=str,
                        default='Quake_10x_Mammary_Gland')
    g_cross.add_argument('--cross_dataset_col', type=str, default='dataset_name')
    g_cross.add_argument('--cross_label_col', type=str, default='cell_ontology_class')
    g_cross.add_argument(
        '--cross_standardization',
        choices=['joint', 'per_dataset'],
        default='joint',
        help='joint matches the manuscript (gene-wise z-score on ref+tgt union); '
             'per_dataset is an explicit target-specific sensitivity analysis'
    )

    args = parser.parse_args()

    if not os.path.exists(args.ontology_path):
        print(f"[Error] Ontology not found at '{args.ontology_path}'. "
              f"Run compile_ontology.py first.")
    else:
        train(args)
