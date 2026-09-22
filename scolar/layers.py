"""Loss functions used by scOLAR: reconstruction, prototype classification, and hierarchical regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """Numerically stable zero-inflated negative-binomial reconstruction loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, mean, disp, drop, sf=None):

        if sf is not None:

            sf = sf.clamp(0.1, 10.0)
            mean = mean * sf

        mean = mean.clamp(min=1e-6, max=1e4)
        theta = disp + self.eps
        theta = theta.clamp(min=1e-4, max=1e4)
        nb_p = mean / (mean + theta + self.eps)

        # --- NB log-likelihood ---
        log_nb = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
            + theta * torch.log(1.0 - nb_p + self.eps)
            + x * torch.log(nb_p + self.eps)
        )
        log_nb = log_nb.clamp(min=-300, max=300)

        drop = drop.clamp(1e-6, 1.0 - 1e-6)
        log_pi = torch.log(drop + self.eps)
        log_1_pi = torch.log1p(-drop + self.eps)

        zero_ll = torch.logsumexp(torch.stack([log_pi, log_1_pi + log_nb], dim=-1), dim=-1)
        nonzero_ll = log_1_pi + log_nb

        zero_mask = (x < self.eps).float()
        ll = zero_mask * zero_ll + (1.0 - zero_mask) * nonzero_ll

        return -ll.mean()


class PrototypeLoss(nn.Module):
    """Cross-entropy loss over ontology-indexed cosine-similarity logits."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,  # [B, N]
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if logits.size(0) == 0:
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()

        return F.cross_entropy(logits / self.temperature, labels)


class HierarchicalCentripetalLoss(nn.Module):
    """Margin-ranking loss for ontology-aware prototype triplets."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prototypes: torch.Tensor,
        triplets: torch.Tensor,
        margin: float = 0.15,
    ) -> torch.Tensor:

        if triplets is None or triplets.numel() == 0:
            return torch.zeros(1, device=prototypes.device, requires_grad=True).squeeze()

        anc = prototypes[triplets[:, 0]]  # [S, D]
        pos = prototypes[triplets[:, 1]]  # [S, D]
        neg = prototypes[triplets[:, 2]]  # [S, D]

        sim_ap = F.cosine_similarity(anc, pos, dim=1)
        sim_an = F.cosine_similarity(anc, neg, dim=1)  # [S]

        loss = F.relu(sim_an - sim_ap + margin)  # [S]
        return loss.mean()


class scOLARLoss(nn.Module):
    """Combine prototype classification, reconstruction, and hierarchical losses."""

    def __init__(
        self,
        ontology_index,
        rec_type: str = "zinb",
        rec_weight: float = 1.0,
        hcl_weight: float = 0.0,
        proto_weight: float = 1.0,
        hcl_margin_start: float = 0.3,
        hcl_margin_end: float = 0.05,
    ):
        super().__init__()

        self.zinb = ZINBLoss()
        self.proto = PrototypeLoss(temperature=1.0)
        self.hcl = HierarchicalCentripetalLoss()

        self.rec_type = rec_type
        self.rec_weight = rec_weight
        self.hcl_weight = hcl_weight
        self.proto_weight = proto_weight

        self.hcl_margin_start = hcl_margin_start
        self.hcl_margin_end = hcl_margin_end

    def _get_hcl_margin(self, epoch: int, total_epochs: int) -> float:
        """Linearly interpolate the hierarchical margin across training epochs."""
        if epoch is None or total_epochs is None or total_epochs <= 1:
            return self.hcl_margin_start
        progress = (epoch - 1) / (total_epochs - 1)
        progress = max(0.0, min(1.0, progress))  # clamp
        return self.hcl_margin_start + progress * (self.hcl_margin_end - self.hcl_margin_start)

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        x_raw: torch.Tensor,
        triplets: torch.Tensor,
        sf: torch.Tensor = None,  # [B, 1] size factor
        mask: torch.Tensor = None,
        epoch: int = None,
        total_epochs: int = None,
        margin_override: float = None,
    ) -> tuple:
        """Return the differentiable total loss and detached scalar components for logging."""

        l_proto = self.proto(outputs["logits_fine"], targets, mask)
        l_proto = l_proto * self.proto_weight

        recon = outputs["recon"]
        if self.rec_type == "zinb":
            l_rec = self.zinb(
                x_raw,
                recon["mean"],
                recon["disp"],
                recon["drop"],
                sf=sf,
            )
        else:

            l_rec = F.mse_loss(recon, x_raw)
        l_rec = l_rec * self.rec_weight

        current_margin = (
            margin_override
            if margin_override is not None
            else self._get_hcl_margin(epoch, total_epochs)
        )

        if "prototypes" not in outputs:
            raise KeyError(
                "[scOLARLoss] 'prototypes' missing from model outputs. "
                "Ensure models.py forward() returns 'prototypes' key."
            )

        protos = F.normalize(outputs["prototypes"], p=2, dim=1)  # [N, D]

        l_hcl = self.hcl(protos, triplets, margin=current_margin)
        l_hcl = l_hcl * self.hcl_weight

        total_loss = l_proto + l_rec + l_hcl

        l_dict = {
            "proto": l_proto.item(),
            "rec": l_rec.item(),
            "hcl": l_hcl.item(),
            "total": total_loss.item(),
        }

        return total_loss, l_dict
