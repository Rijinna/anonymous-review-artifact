"""
hierarchy_triplet_sampling.py — 层级三元组采样
================================================
实现 scOLAR 论文 2.2 的 Hierarchical Centripetal Loss 三元组采样策略：
    - Anchor : 任意细胞类型原型
    - Positive: 与 Anchor 共享至少一个祖先节点（同谱系）
    - Negative: 与 Anchor 不共享祖先节点（不同谱系）

采样策略：
    Hard (75%)  : Positive = 共享祖先数最少的同谱系样本（难正例）
                  Negative = 共享祖先数最少的异谱系样本（难负例）
    Easy (25%)  : 在有效候选中均匀随机采样

已修复的问题（相比旧版）：
  [T1] 设备不一致：ancestor_matrix 在 CPU，indices 在 GPU 时崩溃
       修复：ancestor_matrix 先移到 indices 所在设备再做索引
  [T2] 签名缺少 margin 参数：train.py 传入 margin=current_margin
       导致 TypeError: unexpected keyword argument
       修复：添加 margin 参数（保留供上游调用，本函数内为接口参数）
  [T3] _masked_random_pick 的 Gumbel noise 表达式简化
"""

import torch
from typing import Union


def hierarchy_triplet_aug(
    indices:     torch.Tensor,           # [B] 当前 batch 内各样本的本体全局 ID
    ontology,                            # OntologyManager，提供 ancestor_matrix
    num_samples: int   = None,           # 采样三元组数量（默认 min(512, B*(B-1)//4)）
    hard_ratio:  float = 0.75,           # Hard mining 比例
    margin:      float = None,           # [T2 修复] 接受 train.py 传入的 margin（当前由
                                         # scOLARsLoss._get_hcl_margin 统一管理，
                                         # 此处保留接口供未来 semi-hard mining 使用）
    seed:        int   = None,
    device: Union[str, torch.device] = 'cpu',
) -> torch.Tensor:
    """
    对 batch 内的样本，基于 Cell Ontology 层级结构采样 HCL 三元组。

    返回值：
        triplets: [S, 3] long Tensor，每行 = (anchor_global_id, pos_global_id, neg_global_id)
                  S = 过滤后的有效三元组数量（≤ num_samples）
                  空 batch 或无法构成三元组时返回 shape (0, 3) 的空 tensor。

    关于 margin 参数：
        当前版本中 margin 不参与三元组采样逻辑（采样基于拓扑结构而非当前损失值）。
        未来可扩展为 semi-hard mining：只保留当前 margin 违反的三元组，
        需要额外传入 prototype 矩阵计算相似度。此参数保留接口以兼容 train.py。
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ── 入口防御 ────────────────────────────────────────────────
    if indices.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    # indices 中可能包含 -1（novel 细胞被强制脱钩）
    # 训练时 src 的 sy 不含 -1，tgt 的 tgt_pseudo_y 是 argmax (≥ 0)，理论上安全
    # 但防御性检查：过滤掉 -1
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

    # ── [T1 修复] 设备对齐 ───────────────────────────────────────
    # ancestor_matrix 加载时在 ontology.device（通常 CPU）
    # indices 在训练时在 GPU
    # 解决方案：将 ancestor_matrix 临时移到 indices 所在设备做索引
    # 不修改 ontology 内部状态（不 .to_() in-place），仅临时搬运
    anc_matrix = ontology.ancestor_matrix.to(indices.device).float()  # [N, N]，临时在目标设备

    # ── 向量化准备 ───────────────────────────────────────────────
    # batch_anc[i, j] = 1.0 若节点 j 是 indices[i] 的祖先（含自身）
    batch_anc    = anc_matrix[indices]                  # [B, N]
    # shared_count[i, k] = |ancestors(i) ∩ ancestors(k)|（含自身）
    # 数学：shared_count[i,k] = d_LCA(i,k) + 1，其中 d_LCA 为最近公共祖先深度。
    # 诊断结论（见 diagnose_hcl_pool.py）：
    #   shared_count >= 1 ↔ LCA_depth >= 0：所有对成立（退化），原始 bug。
    #   shared_count >= 2 ↔ LCA_depth >= 1：Quake_10x/Smart-seq2 仍 ~95% 正例，仍偏高。
    #   shared_count >= 3 ↔ LCA_depth >= 2：各数据集正例比例 20–79%，负例池充足。
    shared_count = torch.mm(batch_anc, batch_anc.t())   # [B, B]
    shared       = shared_count >= 3                    # [B, B] bool：同谱系掩码（LCA depth ≥ 2，排除仅共享 root 或 depth-1 祖先的对）
    shared.fill_diagonal_(False)                        # 排除自身
    shared_count.fill_diagonal_(0)

    # ── 全向量化采样 ─────────────────────────────────────────────
    anchor_ids = torch.randint(0, B, (num_samples,), device=device)   # [S]

    anc_counts = shared_count[anchor_ids]   # [S, B] 每个 anchor 与所有样本的共享祖先数
    anc_shared = shared[anchor_ids]         # [S, B] bool

    # Hard Positive：同谱系（shared=True）中共享祖先最少（最难）的 top-k
    # 共享祖先少 = 在 ontology DAG 里只有高层粗粒度祖先相同，是更难的正例
    pos_scores = anc_counts.clone()
    pos_scores[~anc_shared] = float('inf')                              # 屏蔽非正候选
    k_pos = min(3, B)
    hard_pos_pool = torch.topk(pos_scores, k=k_pos, largest=False).indices   # [S, k_pos]

    # Hard Negative：不同谱系（shared=False）中共享祖先最多（最难）的 top-k
    # 共享祖先多但不同谱系 = 视觉上"容易混淆"的难负例（跨谱系但有远亲）
    neg_scores = anc_counts.clone()
    neg_scores[anc_shared]   = float('inf')                             # 屏蔽同谱系
    neg_scores.diagonal().fill_(float('inf'))                           # 屏蔽自身
    k_neg = min(5, B)
    hard_neg_pool = torch.topk(neg_scores, k=k_neg, largest=False).indices   # [S, k_neg]

    # 混合采样：hard_ratio 比例走 hard mining，其余均匀随机
    use_hard = torch.rand(num_samples, device=device) < hard_ratio      # [S] bool

    # Hard 路径：从 pool 中随机选一个
    pos_pick = torch.randint(0, k_pos, (num_samples,), device=device)
    neg_pick = torch.randint(0, k_neg, (num_samples,), device=device)
    pos_hard = hard_pos_pool[torch.arange(num_samples, device=device), pos_pick]
    neg_hard = hard_neg_pool[torch.arange(num_samples, device=device), neg_pick]

    # Easy 路径：在有效候选中均匀随机（Gumbel trick 向量化）
    pos_easy = _gumbel_pick(anc_shared,  device)    # [S]
    neg_easy = _gumbel_pick(~anc_shared, device)    # [S]

    # 合并两条路径
    pos_local = torch.where(use_hard, pos_hard, pos_easy)   # [S] batch-local index
    neg_local = torch.where(use_hard, neg_hard, neg_easy)   # [S]

    # ── 过滤无效三元组 ────────────────────────────────────────────
    # 当某 anchor 的整个 batch 内没有有效正/负候选时，其三元组无效
    row_idx = torch.arange(num_samples, device=device)
    pos_valid = pos_scores[row_idx, pos_local] < float('inf')
    neg_valid = neg_scores[row_idx, neg_local] < float('inf')
    valid     = pos_valid & neg_valid

    anchor_ids = anchor_ids[valid]
    pos_local  = pos_local[valid]
    neg_local  = neg_local[valid]

    if len(anchor_ids) == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    # ── 转换为全局本体 ID ─────────────────────────────────────────
    # HierarchicalCentripetalLoss 用全局 ID 索引完整 prototype 矩阵 [N, D]
    triplets = torch.stack([
        indices[anchor_ids],
        indices[pos_local],
        indices[neg_local],
    ], dim=1)                                           # [S, 3]

    # 去重（同一三元组可能被多次采样）
    return torch.unique(triplets, dim=0)


def _gumbel_pick(
    mask:   torch.Tensor,   # [S, B] bool，True = 有效候选
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    对 [S, B] bool mask，每行均匀随机选一个 True 位置。

    使用 Gumbel-Max trick：
        对 True 位置加 Gumbel(0,1) 噪声并取 argmax，等价于均匀随机采样。
        False 位置填 -inf，保证不会被选中。

    Gumbel(0,1) 采样：若 U ~ Uniform(0,1)，则 -log(-log(U)) ~ Gumbel(0,1)。
    """
    # [T3] 简化：从两次 .log().neg() 改为显式公式
    U     = torch.rand(mask.shape, device=device).clamp(min=1e-10)  # 防 log(0)
    noise = -torch.log(-torch.log(U))                                # Gumbel(0,1)
    noise[~mask] = -float('inf')                                     # 屏蔽无效位置
    return noise.argmax(dim=1)                                        # [S]


# =============================================================================
# 辅助：scRNA 稀疏 Dropout 增强（baseline scBOL 保留）
# =============================================================================

def dropout_aug(X: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """
    模拟 scRNA-seq 技术 dropout 的稀疏增强。
    随机将部分基因表达值置 0，增强 encoder 对稀疏性的鲁棒性。
    """
    mask = torch.rand(X.shape, device=X.device) > dropout_rate
    return X * mask


if __name__ == "__main__":
    print("hierarchy_triplet_aug (scBOL+ v0.4) ready.")