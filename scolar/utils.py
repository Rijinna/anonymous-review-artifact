import torch
import numpy as np
import scipy.sparse
import tqdm
import unicodedata
import collections


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def norm(s):
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip().casefold()


def in_ipynb():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


class OntologyManager:
    """Load compiled Cell Ontology tensors and expose deterministic label and ancestry utilities."""

    def __init__(self, compiled_path, device="cpu", coarse_depth_threshold=3):
        """Initialize ontology mappings, ancestry tensors, depths, and coarse memberships."""
        print(f"[Utils] Loading compiled ontology from {compiled_path}...")
        checkpoint = torch.load(compiled_path, map_location=device)

        self.device = device
        self.id2idx = checkpoint["mappings"]["id2idx"]
        self.term_map = checkpoint["mappings"]["term_map"]
        self.all_terms = checkpoint["structure"]["all_terms"]

        self.ancestor_matrix = (
            checkpoint["structure"]["ancestor_matrix_incl_self"].to(device).bool()
        )
        self.depths = checkpoint["structure"]["depths"].to(device)  # [N]

        self.num_classes = len(self.all_terms)

        self.fine_to_coarse_mask, self.coarse_node_ids = self._build_fine_to_coarse_mask(
            coarse_depth_threshold
        )

        self.num_coarse_classes = len(self.coarse_node_ids)

        print(
            f"[Utils] Ontology loaded: {self.num_classes} nodes total, "
            f"{self.num_coarse_classes} coarse nodes (depth ≤ {coarse_depth_threshold})"
        )

        self._build_id2name()

    def _build_id2name(self):
        """Build a readable reverse mapping from ontology indices to preferred labels."""
        if hasattr(self, "id2name") and self.id2name is not None:
            return

        print("[OntologyManager] Building id2name with readable name priority...")

        id2name = {}
        for name, idx in self.term_map.items():
            if idx not in id2name or (
                str(id2name[idx]).upper().startswith("CL:")
                and not str(name).upper().startswith("CL:")
            ):
                id2name[idx] = name

        self.id2name = id2name
        print(f"[OntologyManager] id2name built successfully: {len(id2name)} entries")

    def _build_fine_to_coarse_mask(self, coarse_depth_threshold: int):
        """Map every ontology term to eligible coarse ancestors within the configured depth."""
        depths_cpu = self.depths.cpu()
        N = self.ancestor_matrix.shape[0]

        coarse_bool_mask = (depths_cpu > 0) & (depths_cpu <= coarse_depth_threshold)
        coarse_ids = torch.where(coarse_bool_mask)[0]

        if len(coarse_ids) == 0:
            raise RuntimeError(
                f"[Utils] No coarse nodes found with depth ≤ {coarse_depth_threshold}. "
                f"Try increasing coarse_depth_threshold. "
                f"Depth distribution: min={int(depths_cpu.min())}, "
                f"max={int(depths_cpu.max())}, "
                f"median={float(depths_cpu.float().median()):.1f}"
            )

        anc = self.ancestor_matrix.cpu()

        # fine_to_coarse[i, k] = anc[i, coarse_ids[k]]

        fine_to_coarse = anc[:, coarse_ids]

        C_coarse = len(coarse_ids)
        k_local = torch.arange(C_coarse)  # [0, 1, ..., C_coarse-1]
        self_mask = torch.zeros(N, C_coarse, dtype=torch.bool)

        valid = coarse_ids < N
        self_mask[coarse_ids[valid], k_local[valid]] = True

        fine_to_coarse = fine_to_coarse & ~self_mask

        has_children = fine_to_coarse.any(dim=0)  # [C_coarse] bool
        n_empty = (~has_children).sum().item()
        if n_empty > 0:
            print(
                f"[Utils] Warning: {n_empty}/{C_coarse} coarse nodes have no fine children "
                f"(they may be leaf nodes at depth ≤ {coarse_depth_threshold}). "
                f"Consider adjusting coarse_depth_threshold."
            )

        return fine_to_coarse.to(self.device), coarse_ids.to(self.device)

    def map_labels(self, labels):
        """Map input labels to ontology indices and return an explicit validity mask."""
        raw_indices = [self.term_map.get(norm(l), -1) for l in labels]
        indices = torch.tensor(raw_indices, dtype=torch.long, device=self.device)
        mask = (indices != -1).to(self.device)

        clean_indices = indices.clone()
        clean_indices[~mask] = 0
        return clean_indices, mask

    def get_ancestor_mask(self, batch_labels, mask=None):
        if mask is not None:
            batch_labels = batch_labels[mask]
        return self.ancestor_matrix[batch_labels]

    def get_hierarchy_weights(self):
        """Return normalized log-depth weights for ontology terms."""
        d = self.depths.clamp(min=0).float()
        w = torch.log1p(d)
        return w / w.max()

    def get_hierarchy_margins(self, base_margin=0.1, max_margin=0.5):
        """Return depth-dependent margins bounded by the requested maximum."""
        d = torch.log1p(self.depths.clamp(min=0).float())
        d = d / d.max()
        return torch.clamp(base_margin * (1 + d), max=max_margin)

    def get_fine_to_coarse_for_src(self, src_ids: torch.Tensor):
        """Return the fine-to-coarse membership rows selected by source ontology indices."""
        src_ids_cpu = src_ids.cpu()
        return self.fine_to_coarse_mask[src_ids_cpu].to(self.device)


# DataDict


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        if self.size == 0:
            return self
        shuffled = DataDict()
        idx = random_state.permutation(self.size)
        for k, v in self.items():
            shuffled[k] = v[idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray, torch.Tensor)):
            return DataDict([(k, v[fetch]) for k, v in self.items()])
        return super().__getitem__(fetch)

    def get_batch(self, batch_size, start_idx=0, device="cpu"):
        end_idx = min(start_idx + batch_size, self.size)
        fetch = slice(start_idx, end_idx)
        batch_data = self[fetch]
        if "X" in batch_data:
            batch_data["X"] = torch.tensor(batch_data["X"], dtype=torch.float, device=device)
        return batch_data


def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)
