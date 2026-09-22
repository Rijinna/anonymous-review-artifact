import torch
import torch.nn as nn
import torch.nn.functional as F


class scOLAR(nn.Module):
    """
    scOLAR 主干模型 (论文 3.1 & 3.2 & 4.1 & 4.3)

    设计原则（对齐论文）：
    ┌─────────────────────────────────────────────────────────────────┐
    │  Input x                                                        │
    │     │                                                           │
    │  Encoder f_φ  →  z  (latent embedding)                        │
    │     │                    │                                      │
    │  Decoder g_ψ         Prototype Head P                          │
    │  (重构路，辅助稳健性)  (分类路，细粒度 logits)                    │
    │     │                    │                                      │
    │  L_recon             L_CE + L_HCL                              │
    │                          │                                      │
    │              Coarse Prototype Construction                      │
    │              (由 P detach 聚合，物理梯度隔离)                    │
    │                          │                                      │
    │                      S_coarse(x)  → L_dec-adv                  │
    └─────────────────────────────────────────────────────────────────┘

    相比旧版的关键修复：
    1. [架构] fine_to_coarse_mask & src_ids 注册为 buffer（register_buffer），
       随 model.to(device) 自动迁移，不再每次 forward 传参。
       旧版依赖 `ontology` 参数，但 train.py 从不传入，导致死代码路径。

    2. [正确性] Coarse Prototype 聚合只使用 src_ids 对应的行。
       旧版用全部 N 行（9127 个节点），大量未训练的 prototype 行会污染聚合结果。
       新版：p_src = p_norm[src_ids].detach()，仅聚合训练集实际出现的 fine 类。

    3. [Bug Fix] ZINB 输出 key 统一为 'disp' / 'drop'。
       旧版输出 'dispersion' / 'dropout'，但 train.py 里消费时用 'disp' / 'drop'，
       导致 KeyError 在运行时静默变成 NaN（取决于 criterion 内部处理方式）。

    4. [接口] forward() 不再接受 ontology 参数（避免歧义），
       通过 set_ontology_info() 一次性注册静态矩阵。
    """

    def __init__(
        self,
        input_dim:   int,
        num_classes: int,       # = ontology.num_classes（全局本体节点总数 N）
        latent_dim:  int  = 128,
        hidden_dim:  int  = 512,
        dropout:     float = 0.2,
        use_zinb:    bool  = False,
    ):
        super().__init__()

        self.use_zinb    = use_zinb
        self.num_classes = num_classes  # N，全局本体空间大小

        # ── 共享 Encoder f_φ ───────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ── 重构路径 Decoder g_ψ ───────────────────────────────────
        self.decoder_base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        if self.use_zinb:
            # ZINB 三个输出头（对应负二项分布的均值、离散参数、零膨胀概率）
            self.recon_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softmax(dim=-1))
            self.recon_disp = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softplus())
            self.recon_drop = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        else:
            self.recon_head = nn.Linear(hidden_dim, input_dim)

        # ── Fine-level Prototype Head P ────────────────────────────
        # shape: [N, latent_dim]，N = 全局本体节点总数
        # 训练中只有 src_ids 对应的行会收到梯度；其他行保持 Xavier 初始化值。
        self.prototype_head = nn.Parameter(torch.Tensor(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.prototype_head)

        # 可学习温度系数（初始值 8.0 对应 cosine similarity 常用缩放范围）
        self.scale = nn.Parameter(torch.tensor(8.0))

        # ── 静态 Ontology 信息（由 set_ontology_info 注册）──────────
        # 使用 register_buffer：
        #   - 不是可训练参数（不进入 optimizer）
        #   - 随 model.to(device) / model.cuda() 自动迁移
        #   - 保存在 state_dict 里（checkpoint 可复现）
        #
        # 初始化为 None，训练前必须调用 set_ontology_info()
        self.register_buffer('fine_to_coarse_mask', None)   # [N, C_coarse] bool
        self.register_buffer('src_ids',             None)   # [n_src] long

    # ──────────────────────────────────────────────────────────────
    # 公共接口：注册静态 Ontology 信息（训练前调用一次）
    # ──────────────────────────────────────────────────────────────
    def set_ontology_info(
        self,
        fine_to_coarse_mask: torch.Tensor,  # [N, C_coarse] bool，来自 OntologyManager
        src_ids:             torch.Tensor,  # [n_src] long，来自 prepare_datasets_robust
    ):
        """
        注册静态本体信息，供 forward 中 Coarse Prototype 构建使用。

        调用时机（在 train.py 中）：
            src_loader, ..., unique_src_ids = prepare_datasets_robust(args, ontology)
            model.set_ontology_info(ontology.fine_to_coarse_mask, unique_src_ids)

        为什么用 copy_() 而不是直接赋值：
            register_buffer 注册的 None 在 .to(device) 后仍是 None，
            直接赋值会绕过 buffer 机制（变成普通属性，不随 .to(device) 迁移）。
            用 _set_buffer 辅助方法保证正确注册。
        """
        # 必须先 register 再赋值才能让 buffer 生效
        # 这里直接重新 register（覆盖初始 None）
        self.register_buffer('fine_to_coarse_mask', fine_to_coarse_mask.bool())
        self.register_buffer('src_ids',             src_ids.long())

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, input_dim] 输入特征（经过 normalize + log1p）

        Returns:
            dict with keys:
              'z'            : [B, latent_dim]  latent embedding
              'recon'        : dict {'mean', 'disp', 'drop'} (ZINB) 或 Tensor (MSE)
              'logits_fine'  : [B, N]           细粒度 logit（未归一化余弦相似度 × scale）
              'prototypes'   : [N, latent_dim]  raw prototype 参数（供 HCL 使用）
              's_coarse_max' : [B]              S_coarse(x)（仅当 ontology info 已注册时存在）
        """
        # ── 1. 特征提取 ────────────────────────────────────────────
        z = self.encoder(x)                           # [B, latent_dim]

        # ── 2. 重构路径 ────────────────────────────────────────────
        h_recon = self.decoder_base(z)                # [B, hidden_dim]
        if self.use_zinb:
            recon = {
                'mean': self.recon_mean(h_recon),     # [B, input_dim]，Softmax 归一化
                'disp': self.recon_disp(h_recon),     # [B, input_dim]，Softplus > 0
                'drop': self.recon_drop(h_recon),     # [B, input_dim]，Sigmoid ∈ (0,1)
            }
        else:
            recon = self.recon_head(h_recon)          # [B, input_dim]

        # ── 3. 分类路径：Fine-level Logits ─────────────────────────
        z_norm = F.normalize(z, p=2, dim=1)                          # [B, latent_dim]
        p_norm = F.normalize(self.prototype_head, p=2, dim=1)        # [N, latent_dim]
        logits_fine = self.scale * torch.mm(z_norm, p_norm.t())      # [B, N]

        outputs = {
            'z':           z,
            'recon':       recon,
            'logits_fine': logits_fine,
            'prototypes':  self.prototype_head,   # raw（非归一化），供 HCL triplet 使用
        }

        # ── 4. Coarse Prototype 构建 & S_coarse(x) 计算 ────────────
        #
        # 论文 4.1：Coarse Prototype 仅作几何参考，不参与决策路径。
        #          因此必须对 fine prototype 做梯度隔离（.detach()），
        #          确保 L_dec-adv 的梯度不流回 prototype_head。
        #
        # 论文 4.3：S_coarse(x) = max_c s(z, μ_c^coarse)
        #          只用于外层 criterion 计算 L_dec-adv，不参与分类。
        #
        # 实现细节：
        #   - 只聚合 src_ids 对应行（训练集实际出现的 fine 类）
        #   - fine_to_coarse_mask[src_ids]: [n_src, C_coarse]
        #     表示每个 src fine 节点属于哪些 coarse 父节点
        #   - agg_weight = mask 归一化 → 均匀平均（等权聚合）
        #
        if self.fine_to_coarse_mask is not None and self.src_ids is not None:
            outputs['s_coarse_max'] = self._compute_s_coarse(z_norm, p_norm)

        return outputs

    def _compute_s_coarse(
        self,
        z_norm: torch.Tensor,   # [B, latent_dim]，已归一化，有梯度
        p_norm: torch.Tensor,   # [N, latent_dim]，已归一化，有梯度
    ) -> torch.Tensor:
        """
        计算 S_coarse(x) = max_c s(z, μ_c^coarse)。

        独立提取为方法，便于单元测试与后续调试。

        Step A: 从 prototype_head 中提取 src fine prototypes，并 detach。
                p_src = p_norm[src_ids].detach()    [n_src, latent_dim]
                detach() 是物理级梯度隔离，比 .data 更安全（仍在 autograd 图外）。

        Step B: 构建聚合权重矩阵。
                sub_mask = fine_to_coarse_mask[src_ids]    [n_src, C_coarse] bool
                转 float 后按列归一化（每列 = 每个 coarse 节点的子节点集合）：
                agg_w[k] = sub_mask[:, k] / sub_mask[:, k].sum()   等权平均

        Step C: 聚合得到 Coarse Prototype。
                p_coarse_raw = agg_w.t() @ p_src          [C_coarse, latent_dim]
                重新 L2 归一化 → p_coarse                  [C_coarse, latent_dim]

        Step D: 计算余弦相似度并取 max。
                sim_coarse = z_norm @ p_coarse.t()         [B, C_coarse]
                s_coarse_max = sim_coarse.max(dim=1).values [B]

        Returns:
            s_coarse_max: [B] float Tensor，在 z 上有梯度，在 prototype_head 上无梯度。
        """
        device = z_norm.device

        # Step A: 提取并 detach src fine prototypes
        src_ids = self.src_ids                                    # [n_src]
        p_src   = p_norm[src_ids].detach()                        # [n_src, latent_dim]

        # Step B: 聚合权重矩阵
        sub_mask = self.fine_to_coarse_mask[src_ids].float()      # [n_src, C_coarse]
        col_sums = sub_mask.sum(dim=0, keepdim=True).clamp(min=1) # [1, C_coarse]，防除零
        agg_w    = sub_mask / col_sums                            # [n_src, C_coarse]，列归一化

        # Step C: 聚合 → 归一化
        p_coarse_raw = agg_w.t() @ p_src                          # [C_coarse, latent_dim]
        p_coarse     = F.normalize(p_coarse_raw, p=2, dim=1)      # [C_coarse, latent_dim]

        # Step D: 余弦相似度 → max
        sim_coarse   = z_norm @ p_coarse.t()                      # [B, C_coarse]
        s_coarse_max = sim_coarse.max(dim=1).values                # [B]

        return s_coarse_max

    # ──────────────────────────────────────────────────────────────
    # 便捷访问
    # ──────────────────────────────────────────────────────────────
    def get_prototypes(self) -> torch.Tensor:
        """
        返回 L2 归一化后的 prototype（[N, latent_dim]），供 HCL triplet loss 使用。
        注意：返回的 tensor 仍在计算图中（有梯度），外部如需隔离请自行 .detach()。
        """
        return F.normalize(self.prototype_head, p=2, dim=1)