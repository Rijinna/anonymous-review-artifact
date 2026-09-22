"""
layers.py — scOLAR 损失函数模块
=================================
包含：
  ZINBLoss                  : 负二项分布 + 零膨胀重构损失（对齐 scVI 标准）
  PrototypeLoss             : Fine-level 原型分类损失（论文 3.2）
  HierarchicalCentripetalLoss: 层级向心损失（论文 2.2，HCL）
  scOLARLoss                : 统一损失管理器

已修复的问题（相比旧版）：
  [L1] SyntaxError：recon['drop'] 后缺少逗号
  [L2] scOLARLoss.forward 签名缺少 sf 参数 → TypeError: unexpected keyword 'sf'
  [L3] HCL margin 双重调度：外部 current_margin 通过 train.py 控制，
       内部 margin_factor 独立衰减，两者叠加逻辑混乱。
       修复：HCL 接受外部传入的 margin（float）作为当前有效 margin，
       由 scOLARLoss 统一调度并传下去。内部 margin_factor 保留
       作为乘数（归一化因子），但 base_margin 改为外部注入。
  [L4] ZINBLoss 参数名统一为 disp/drop，与 models.py 输出 key 对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. ZINB 重构损失（论文 3.1 Denoising Reconstruction）
# =============================================================================

class ZINBLoss(nn.Module):
    """
    数值稳定的 Zero-Inflated Negative Binomial 损失。

    严格遵循 scVI (Lopez et al. 2018) 的参数化方式：
        p(x | z) = π · δ(x=0) + (1-π) · NB(x; μ, θ)

    其中：
        μ (mean)  : Decoder Softmax 输出，代表各基因的相对表达量
        θ (disp)  : Softplus 输出，每基因离散参数（> 0）
        π (drop)  : Sigmoid 输出，零膨胀概率 ∈ (0, 1)
        sf        : 文库 Size Factor，将相对 μ 缩放回绝对 count 空间

    数值稳定性措施：
        - NB 的成功概率 p = μ / (μ + θ) 在 log 空间计算
        - zero-inflation 混合项用 logsumexp，避免概率截断
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, mean, disp, drop, sf=None):

        # --- Size Factor 缩放 ---
        if sf is not None:
            # sf clip 防止极端测序深度
            sf = sf.clamp(0.1, 10.0)
            mean = mean * sf

        # --- NB 参数 ---
        # mean clamp 防止 lgamma 溢出
        mean  = mean.clamp(min=1e-6, max=1e4)       # ← 新增
        theta = disp + self.eps
        theta = theta.clamp(min=1e-4, max=1e4)      # ← 新增，防止 theta 趋近 0
        nb_p  = mean / (mean + theta + self.eps)

        # --- NB log-likelihood ---
        log_nb = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
            + theta * torch.log(1.0 - nb_p + self.eps)
            + x     * torch.log(nb_p      + self.eps)
        )
        log_nb = log_nb.clamp(min=-300, max=300)    # ← 新增，防止 lgamma 极端值传播

        # --- Zero-inflation 混合 ---
        drop = drop.clamp(1e-6, 1.0 - 1e-6)        # ← 新增，防止 sigmoid 完全饱和
        log_pi    = torch.log(drop   + self.eps)
        log_1_pi  = torch.log1p(-drop + self.eps)

        zero_ll   = torch.logsumexp(
            torch.stack([log_pi, log_1_pi + log_nb], dim=-1), dim=-1
        )
        nonzero_ll = log_1_pi + log_nb

        zero_mask = (x < self.eps).float()
        ll = zero_mask * zero_ll + (1.0 - zero_mask) * nonzero_ll

        return -ll.mean()

# =============================================================================
# 2. Prototype 分类损失（论文 3.2）
# =============================================================================

class PrototypeLoss(nn.Module):
    """
    Fine-level Prototype Classification Loss（论文 3.2）。

    输入 logits 已经过 models.py 中的可学习 scale 缩放（cosine similarity × s），
    不再额外除以 temperature（会导致双重缩放）。

    mask 过滤：只对本体映射有效的样本（valid known cells）计算分类损失，
    自动排除 novel 细胞和 ontology 映射失败的样本。
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,        # [B, N]
        labels: torch.Tensor,        # [B]  全局本体 ID（long）
        mask:   torch.Tensor = None, # [B]  bool，True = 有效样本
    ) -> torch.Tensor:

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if logits.size(0) == 0:
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()

        return F.cross_entropy(logits / self.temperature, labels)


# =============================================================================
# 3. 层级向心损失（论文 2.2 HCL）
# =============================================================================

class HierarchicalCentripetalLoss(nn.Module):
    """
    Hierarchical Centripetal Loss（HCL，论文 2.2）。

    核心约束（排序约束）：
        同一粗粒度父类下的 fine prototype 对，其余弦相似度
        应 > 不同父类下 fine prototype 对的余弦相似度 + margin。

    损失形式（Hinge Loss over triplets）：
        L_HCL = E[ ReLU(sim(a,n) - sim(a,p) + margin) ]

    margin 调度（外部注入，论文 F7 改进）：
        由 scOLARLoss 根据当前 epoch 计算 effective_margin，
        作为 `margin` 参数传入此 forward()。
        不在此类内部做 epoch 调度，避免与外部逻辑重叠。

    Args（forward）：
        prototypes : [N, D]  完整 prototype 矩阵（已 L2 归一化）
        triplets   : [S, 3]  全局本体 ID 的三元组 (anchor, pos, neg)
        margin     : float   当前有效 margin（由外部动态调度传入）
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prototypes: torch.Tensor,    # [N, D] L2 归一化后的完整 prototype
        triplets:   torch.Tensor,    # [S, 3] 全局本体 ID
        margin:     float = 0.15,    # 当前有效 margin（外部注入）
    ) -> torch.Tensor:

        if triplets is None or triplets.numel() == 0:
            return torch.zeros(1, device=prototypes.device, requires_grad=True).squeeze()

        # triplets 列：[anchor_id, pos_id, neg_id]，均为全局本体 ID
        anc = prototypes[triplets[:, 0]]   # [S, D]
        pos = prototypes[triplets[:, 1]]   # [S, D]
        neg = prototypes[triplets[:, 2]]   # [S, D]

        sim_ap = F.cosine_similarity(anc, pos, dim=1)   # [S] ∈ [-1, 1]
        sim_an = F.cosine_similarity(anc, neg, dim=1)   # [S]

        # Hinge Loss：当 neg 比 pos 更相似时才惩罚
        loss = F.relu(sim_an - sim_ap + margin)         # [S]
        return loss.mean()


# =============================================================================
# 4. 统一损失管理器（论文 Loss = L_CE + L_HCL + L_dec-adv + γ·L_recon）
# =============================================================================

class scOLARLoss(nn.Module):
    """
    scOLAR 统一损失管理器。

    封装 L_proto、L_recon、L_HCL 三项，通过权重参数支持 ablation。
    L_dec-adv 由 train.py 的 compute_adv_loss 单独计算（因需访问模型 buffer）。

    权重动态调度（由 train.py 在外部修改属性）：
        hcl_weight  : epoch 20 后激活（train.py: criterion.hcl_weight = args.lambda_hcl）
        rec_weight  : epoch 50 后减半（train.py: criterion.rec_weight *= 0.5）

    HCL margin 调度（论文 F7）：
        hcl_margin_start / hcl_margin_end 由 train.py 传入，
        scOLARLoss.forward 根据 epoch 线性插值，注入 HCL.forward。
        这保证了"早期大 margin 粗调，后期小 margin 细调"的逻辑统一在损失模块内。
    """

    def __init__(
        self,
        ontology_index,           # OntologyManager（HCL 子模块持有引用）
        rec_type:     str   = 'zinb',
        rec_weight:   float = 1.0,
        hcl_weight:   float = 0.0,  # 默认关闭，epoch 20 激活
        proto_weight: float = 1.0,
        hcl_margin_start: float = 0.3,   # 早期大 margin（论文 F7）
        hcl_margin_end:   float = 0.05,  # 后期小 margin
    ):
        super().__init__()

        # 子模块
        self.zinb  = ZINBLoss()
        self.proto = PrototypeLoss(temperature=1.0)
        self.hcl   = HierarchicalCentripetalLoss()

        # 权重（train.py 可在外部直接修改）
        self.rec_type    = rec_type
        self.rec_weight  = rec_weight
        self.hcl_weight  = hcl_weight
        self.proto_weight = proto_weight

        # HCL margin 调度参数（外部 margin 注入链路）
        self.hcl_margin_start = hcl_margin_start
        self.hcl_margin_end   = hcl_margin_end

    def _get_hcl_margin(self, epoch: int, total_epochs: int) -> float:
        """
        线性插值计算当前 epoch 的 HCL margin。
        epoch 1  → hcl_margin_start（大 margin，粗调）
        epoch T  → hcl_margin_end（小 margin，细调）
        """
        if epoch is None or total_epochs is None or total_epochs <= 1:
            return self.hcl_margin_start
        progress = (epoch - 1) / (total_epochs - 1)        # 0.0 → 1.0
        progress = max(0.0, min(1.0, progress))             # clamp
        return self.hcl_margin_start + progress * (
            self.hcl_margin_end - self.hcl_margin_start
        )

    def forward(
        self,
        outputs:      dict,            # model.forward() 的完整输出 dict
        targets:      torch.Tensor,    # [B] 全局本体 ID（long）
        x_raw:        torch.Tensor,    # [B, G] 原始 count（ZINB 用）
        triplets:     torch.Tensor,    # [S, 3] 全局本体 ID 三元组
        sf:           torch.Tensor = None,  # [B, 1] size factor  ← [L2 修复]
        mask:         torch.Tensor = None,  # [B] bool 有效样本掩码
        epoch:        int          = None,
        total_epochs: int          = None,
        margin_override: float     = None,
    ) -> tuple:
        """
        Returns:
            total_loss : scalar tensor（有梯度）
            l_dict     : dict，各分项的 float 值（用于日志，.item() 已调用）
        """

        # ── 1. Prototype 分类损失 L_CE ────────────────────────────
        l_proto = self.proto(outputs['logits_fine'], targets, mask)
        l_proto = l_proto * self.proto_weight

        # ── 2. ZINB 重构损失 L_recon（Source）─────────────────────
        recon = outputs['recon']
        if self.rec_type == 'zinb':
            # [L1 修复] 正确传入 sf；[L4 修复] key 名与 models.py 一致
            l_rec = self.zinb(
                x_raw,
                recon['mean'],
                recon['disp'],
                recon['drop'],   # ← [L1] 此处逗号之前缺失，已修复
                sf=sf,
            )
        else:
            # MSE fallback（use_zinb=False 时）
            l_rec = F.mse_loss(recon, x_raw)
        l_rec = l_rec * self.rec_weight

        # ── 3. 层级向心损失 L_HCL ─────────────────────────────────
        # [L3 修复] margin 由本类统一调度并注入 HCL，不在 HCL 内部重复衰减
        current_margin = margin_override if margin_override is not None \
                     else self._get_hcl_margin(epoch, total_epochs)

        if 'prototypes' not in outputs:
            raise KeyError(
                "[scOLARLoss] 'prototypes' missing from model outputs. "
                "Ensure models.py forward() returns 'prototypes' key."
            )
        # L2 归一化（triplet similarity 计算用 cosine，需归一化）
        protos = F.normalize(outputs['prototypes'], p=2, dim=1)  # [N, D]

        l_hcl = self.hcl(protos, triplets, margin=current_margin)
        l_hcl = l_hcl * self.hcl_weight

        # ── 合并总损失 ────────────────────────────────────────────
        total_loss = l_proto + l_rec + l_hcl

        l_dict = {
            'proto': l_proto.item(),
            'rec':   l_rec.item(),
            'hcl':   l_hcl.item(),
            'total': total_loss.item(),
        }

        return total_loss, l_dict

# backward compatibility alias
scOLARLoss = scOLARLoss