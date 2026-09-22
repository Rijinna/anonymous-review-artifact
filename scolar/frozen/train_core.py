import argparse, hashlib, json, math, os, struct, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from utils import OntologyManager
from models import scOLAR
from layers import scOLARLoss
from hierarchy_triplet_sampling import hierarchy_triplet_aug


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

def _finite(x):
    return x is None or torch.isfinite(x).all().item()

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
    configure_loader_rng(src_train_loader, src_eval_loader, tgt_train_loader, tgt_eval_loader)
    record_initial_model(model, args)
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
                raise RuntimeError(f"non-finite forward at epoch {epoch}")

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
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
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

    return model, ontology, alpha, unique_src_ids, calibration_plan, time.time()-start_time


def make_parser():
    parser = argparse.ArgumentParser(
            description="scOLAR Training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
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
    g_model = parser.add_argument_group("Model")
    g_model.add_argument('--use_zinb', action=argparse.BooleanOptionalAction, default=True,
                             help='Use ZINB reconstruction loss')
    g_model.add_argument('--coarse_depth_threshold', type=int, default=3,
                             help='Depth threshold for coarse nodes in OntologyManager')
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
    return parser
