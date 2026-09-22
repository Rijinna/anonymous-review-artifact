import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
def _l2_normalize_embeddings(z) -> np.ndarray:
    """Common external evaluator preprocessing: row-wise L2 normalization."""
    if isinstance(z, torch.Tensor):
        z_t = z.detach().cpu().float()
    else:
        z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    if z_t.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape={tuple(z_t.shape)}")
    return F.normalize(z_t, p=2, dim=1).numpy()

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
