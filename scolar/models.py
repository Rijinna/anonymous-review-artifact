import torch
import torch.nn as nn
import torch.nn.functional as F


class scOLAR(nn.Module):
    """
    scOLAR backbone with a shared encoder, reconstruction decoder, and
    ontology-indexed prototype head.

    ``prototype_head`` has one row for every term in the compiled ontology.
    Fine-level logits are computed against that complete matrix. ``src_ids``
    identifies the reference classes used when constructing detached coarse
    prototypes for the decision-level objective.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_zinb: bool = False,
    ):
        super().__init__()

        self.use_zinb = use_zinb
        self.num_classes = num_classes

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

        self.decoder_base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        if self.use_zinb:

            self.recon_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softmax(dim=-1))
            self.recon_disp = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softplus())
            self.recon_drop = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        else:
            self.recon_head = nn.Linear(hidden_dim, input_dim)

        # Shape: [N, latent_dim], where N is the number of compiled ontology terms.
        # Cross-entropy is evaluated over all N logits; src_ids separately controls
        # which rows contribute to coarse-prototype aggregation.
        self.prototype_head = nn.Parameter(torch.Tensor(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.prototype_head)

        self.scale = nn.Parameter(torch.tensor(8.0))

        self.register_buffer("fine_to_coarse_mask", None)  # [N, C_coarse] bool
        self.register_buffer("src_ids", None)  # [n_src] long

    def set_ontology_info(
        self,
        fine_to_coarse_mask: torch.Tensor,
        src_ids: torch.Tensor,
    ):
        """Register ontology membership and source-class indices as model buffers."""

        self.register_buffer("fine_to_coarse_mask", fine_to_coarse_mask.bool())
        self.register_buffer("src_ids", src_ids.long())

    # Forward

    def forward(self, x: torch.Tensor) -> dict:
        """Encode an expression batch and return reconstruction, prototype logits, and optional coarse similarity."""

        z = self.encoder(x)  # [B, latent_dim]

        h_recon = self.decoder_base(z)  # [B, hidden_dim]
        if self.use_zinb:
            recon = {
                "mean": self.recon_mean(h_recon),
                "disp": self.recon_disp(h_recon),
                "drop": self.recon_drop(h_recon),
            }
        else:
            recon = self.recon_head(h_recon)  # [B, input_dim]

        z_norm = F.normalize(z, p=2, dim=1)  # [B, latent_dim]
        p_norm = F.normalize(self.prototype_head, p=2, dim=1)  # [N, latent_dim]
        logits_fine = self.scale * torch.mm(z_norm, p_norm.t())  # [B, N]

        outputs = {
            "z": z,
            "recon": recon,
            "logits_fine": logits_fine,
            "prototypes": self.prototype_head,
        }

        #   - fine_to_coarse_mask[src_ids]: [n_src, C_coarse]

        if self.fine_to_coarse_mask is not None and self.src_ids is not None:
            outputs["s_coarse_max"] = self._compute_s_coarse(z_norm, p_norm)

        return outputs

    def _compute_s_coarse(
        self,
        z_norm: torch.Tensor,
        p_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute maximum similarity to detached coarse prototypes aggregated from source-class rows."""
        device = z_norm.device

        src_ids = self.src_ids  # [n_src]
        p_src = p_norm[src_ids].detach()  # [n_src, latent_dim]

        sub_mask = self.fine_to_coarse_mask[src_ids].float()  # [n_src, C_coarse]
        col_sums = sub_mask.sum(dim=0, keepdim=True).clamp(min=1)
        agg_w = sub_mask / col_sums

        p_coarse_raw = agg_w.t() @ p_src  # [C_coarse, latent_dim]
        p_coarse = F.normalize(p_coarse_raw, p=2, dim=1)  # [C_coarse, latent_dim]

        sim_coarse = z_norm @ p_coarse.t()  # [B, C_coarse]
        s_coarse_max = sim_coarse.max(dim=1).values  # [B]

        return s_coarse_max

    def get_prototypes(self) -> torch.Tensor:
        """Return L2-normalized ontology-indexed prototypes."""
        return F.normalize(self.prototype_head, p=2, dim=1)
