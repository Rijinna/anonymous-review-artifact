"""Sample ontology-aware triplets for the hierarchical centripetal objective."""

import torch
from typing import Union


def hierarchy_triplet_aug(
    indices: torch.Tensor,
    ontology,
    num_samples: int = None,
    hard_ratio: float = 0.75,
    margin: float = None,
    seed: int = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Sample valid anchor, positive, and negative ontology-term triplets within a batch."""
    if seed is not None:
        torch.manual_seed(seed)

    if indices.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    valid_mask = indices >= 0
    if not valid_mask.all():
        indices = indices[valid_mask]

    B = len(indices)
    if B < 4:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    assert indices.max() < ontology.num_classes, (
        f"[Aug] indices out of range: max={indices.max().item()}, "
        f"num_classes={ontology.num_classes}"
    )

    num_samples = num_samples or min(512, B * (B - 1) // 4)

    anc_matrix = ontology.ancestor_matrix.to(indices.device).float()

    batch_anc = anc_matrix[indices]  # [B, N]

    # Require an LCA depth of at least 2 so that sharing only the root or a
    # depth-1 ancestor does not define a positive pair.
    shared_count = torch.mm(batch_anc, batch_anc.t())  # [B, B]
    shared = shared_count >= 3
    shared.fill_diagonal_(False)
    shared_count.fill_diagonal_(0)

    anchor_ids = torch.randint(0, B, (num_samples,), device=device)  # [S]

    anc_counts = shared_count[anchor_ids]
    anc_shared = shared[anchor_ids]  # [S, B] bool

    pos_scores = anc_counts.clone()
    pos_scores[~anc_shared] = float("inf")
    k_pos = min(3, B)
    hard_pos_pool = torch.topk(pos_scores, k=k_pos, largest=False).indices  # [S, k_pos]

    neg_scores = anc_counts.clone()
    neg_scores[anc_shared] = float("inf")
    neg_scores.diagonal().fill_(float("inf"))
    k_neg = min(5, B)
    hard_neg_pool = torch.topk(neg_scores, k=k_neg, largest=False).indices  # [S, k_neg]

    use_hard = torch.rand(num_samples, device=device) < hard_ratio  # [S] bool

    pos_pick = torch.randint(0, k_pos, (num_samples,), device=device)
    neg_pick = torch.randint(0, k_neg, (num_samples,), device=device)
    pos_hard = hard_pos_pool[torch.arange(num_samples, device=device), pos_pick]
    neg_hard = hard_neg_pool[torch.arange(num_samples, device=device), neg_pick]

    pos_easy = _gumbel_pick(anc_shared, device)  # [S]
    neg_easy = _gumbel_pick(~anc_shared, device)  # [S]

    pos_local = torch.where(use_hard, pos_hard, pos_easy)  # [S] batch-local index
    neg_local = torch.where(use_hard, neg_hard, neg_easy)  # [S]

    row_idx = torch.arange(num_samples, device=device)
    pos_valid = pos_scores[row_idx, pos_local] < float("inf")
    neg_valid = neg_scores[row_idx, neg_local] < float("inf")
    valid = pos_valid & neg_valid

    anchor_ids = anchor_ids[valid]
    pos_local = pos_local[valid]
    neg_local = neg_local[valid]

    if len(anchor_ids) == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    triplets = torch.stack(
        [
            indices[anchor_ids],
            indices[pos_local],
            indices[neg_local],
        ],
        dim=1,
    )  # [S, 3]

    return torch.unique(triplets, dim=0)


def _gumbel_pick(
    mask: torch.Tensor,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Select one valid position per row uniformly with the Gumbel-max trick."""
    U = torch.rand(mask.shape, device=device).clamp(min=1e-10)
    noise = -torch.log(-torch.log(U))  # Gumbel(0,1)
    noise[~mask] = -float("inf")
    return noise.argmax(dim=1)  # [S]


def dropout_aug(X: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """Randomly mask expression values to simulate sparse measurement dropout."""
    mask = torch.rand(X.shape, device=X.device) > dropout_rate
    return X * mask


if __name__ == "__main__":
    print("hierarchy_triplet_aug (scOLAR) ready.")
