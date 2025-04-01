import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import knn
from torch_geometric.utils import remove_self_loops
from einops import einsum


class GraphTotalVariation(nn.Module):
    def __init__(
        self,
        k: int,
        gamma: float | None = 2.0,
        loss_weight: float | None = 1.0,
    ):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(
        self,
        coord: torch.Tensor,
        intensity: torch.Tensor,
        out: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ):
        assert out.size(0) == coord.size(0) == target.size(0) == intensity.size(0)

        out = F.normalize(out)

        with torch.no_grad():
            batch, coord = torch.split(coord, [1, 3], dim=1)
            batch = batch.squeeze(dim=1).contiguous()
            coord = coord.contiguous()

            edges = knn(coord, coord, self.k + 1, batch, batch)  # +1 for self-loops
            edges = remove_self_loops(edges)[0]  # which are removed

            p_i = coord[edges[0, :]].detach()
            p_j = coord[edges[1, :]].detach()
            weight = torch.exp(
                -torch.pow(torch.linalg.norm(p_i - p_j, dim=1, ord=2) / self.gamma, 2)
            )

        out_i = out[edges[0, :]]
        out_j = out[edges[1, :]]

        loss = F.l1_loss(out_i, out_j, reduction="none")

        loss = (loss * weight.unsqueeze(dim=-1)).sum(dim=-1).mean()

        return loss * self.loss_weight


class TemporalGraphTotalVariation(nn.Module):
    def __init__(
        self,
        k: int,
        weight_mode: str = "distance",
        gamma: float | None = 2.0,
        loss_weight: float | None = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.k = k
        self.weight_mode = weight_mode.lower()
        assert self.weight_mode in ["distance", "intensity"]
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        coord: torch.Tensor,
        intensity: torch.Tensor,
        out: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ):
        assert out.size(0) == coord.size(0) == target.size(0) == intensity.size(0)
        assert "untransform_coord" in kwargs

        untransform_coord = kwargs["untransform_coord"]

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float32
        ):  # no mixed precision cuda kernels for knn
            batch, coord = torch.split(coord, [1, 3], dim=1)
            batch = batch.squeeze(dim=1).contiguous()
            coord = coord.contiguous()

            # transform coordinates to the same origin
            coord = self._transform(coord, untransform_coord)

            # swap even and odd batches
            even_mask = batch % 2 == 0
            flipped_batch = batch.clone()
            flipped_batch[even_mask] += 1
            flipped_batch[~even_mask] -= 1
            flipped_batch, sort_indices = torch.sort(flipped_batch)
            flipped_batch = flipped_batch.contiguous()
            flipped_coord = coord[sort_indices].contiguous()

            edges = knn(
                x=coord, y=flipped_coord, k=self.k, batch_x=batch, batch_y=flipped_batch
            )

            if self.weight_mode == "distance":
                p_i = coord[edges[1, :]]
                p_j = flipped_coord[edges[0, :]]
                weight = torch.exp(
                    -torch.pow(torch.linalg.norm(p_i - p_j, dim=1) / self.gamma, 2)
                )
            elif self.weight_mode == "intensity":
                assert False, "Not tested."
                # intensity_i = intensity[edges[0, :]]
                # intensity_j = intensity[edges[1, :]]
                # weight = torch.exp(
                #     -torch.pow((intensity_i - intensity_j).abs() / self.gamma, 2)
                # )
            else:
                raise ValueError(f"Unknown {self.weight_mode} weight mode.")

        out = F.normalize(out)

        # transform outputs to the same origin but ignore translation part
        untransform_coord[:, :3, -1] *= 0.0
        out = self._transform(out, untransform_coord)

        out_i = out[edges[1, :]]
        out_j = out[sort_indices][edges[0, :]]

        loss = F.l1_loss(out_i, out_j, reduction="none")
        loss = (loss * weight.unsqueeze(dim=-1)).sum(dim=1).mean()

        return loss * self.loss_weight

    @staticmethod
    def _transform(v, T):
        # v: Nx3
        # T: Nx4x4
        v_homo = torch.concatenate([v, torch.ones_like(v[:, 0:1])], dim=1)
        res = einsum(T, v_homo, "i k j, i j -> i k")
        res = res / res[:, 3:4]
        return res[:, :3]


class L1Loss(nn.Module):
    def __init__(self, loss_weight: float | None = 1.0, reduction: str = "mean"):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        coord: torch.Tensor,
        intensity: torch.Tensor,
        out: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ):
        assert out.size(0) == coord.size(0) == target.size(0)
        assert out.shape == target.shape
        assert "weight" in kwargs
        weight = kwargs["weight"]
        out_normalized = F.normalize(out, p=2, dim=1)
        loss = (
            F.l1_loss(out_normalized, target, reduction="none")
            * weight.unsqueeze(dim=-1)
        ).sum() / weight.sum()
        return loss


class EikonalLoss(nn.Module):
    def __init__(self, loss_weight: float | None = 1.0, reduction: str = "mean"):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        coord: torch.Tensor,
        intensity: torch.Tensor,
        out: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ):
        assert out.ndim == 2
        pred_norm = torch.linalg.norm(out, ord=2, dim=1)
        gt = pred_norm.new_ones(pred_norm.shape)
        loss = F.mse_loss(pred_norm, gt, reduction=self.reduction)
        return loss
