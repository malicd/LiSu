import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.utilities.distributed import gather_all_tensors
import lightning as L
import logging
from tqdm import tqdm

from .PointTransformerV3.model import PointTransformerV3 as PTV3
from .plm import PseudoLabelManager as PLM


class PointTransformerV3(L.LightningModule):
    def __init__(
        self,
        backbone_config: dict,
        head: list[int],
        criteria: list[nn.Module],
        optimizer_cfg: dict,
        lr_scheduler_cfg: dict,
        weights_from_checkpoint_path: str | None = None,
        plm: PLM | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.console_logger = logging.getLogger("lightning.pytorch.core")

        self.optimizer_cfg = optimizer_cfg
        self.optimizer_cfg["lr"] = float(self.optimizer_cfg["lr"])
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.criteria = criteria
        self.weights_from_checkpoint_path = weights_from_checkpoint_path

        self.grid_size = backbone_config.pop("grid_size")
        # PointTransformerV3 requires tuple, not lists
        touple_keys = [
            "stride",
            "enc_depths",
            "enc_channels",
            "enc_num_head",
            "enc_patch_size",
            "dec_depths",
            "dec_channels",
            "dec_num_head",
            "dec_patch_size",
            "pdnorm_conditions",
        ]
        for k in backbone_config:
            if k in touple_keys:
                backbone_config[k] = tuple(backbone_config[k])
        self.backbone = PTV3(**backbone_config)

        self.head = []
        for i in range(len(head) - 2):
            self.head.append(nn.Linear(head[i], head[i + 1]))
            self.head.append(nn.ReLU())
        self.head.append(nn.Linear(head[-2], head[-1], bias=False))
        self.head = nn.Sequential(*self.head)

        self.plm = plm

        # load checkpoint
        if self.weights_from_checkpoint_path is not None:
            checkpoint = torch.load(self.weights_from_checkpoint_path)
            state_dict = checkpoint["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=True
            )
            self.console_logger.info(
                f"Initializing parameters from {self.weights_from_checkpoint_path} finished. "
                f"Missing keys are {missing_keys}, and unexpected keys are {unexpected_keys}."
            )

    def configure_callbacks(self):
        cbs = super().configure_callbacks()
        if self.plm is not None and self.local_rank == 0:  # perform only once per node
            self.plm.configure_callbacks()
        return cbs

    def configure_optimizers(self):
        # adapted from https://github.com/Pointcept/Pointcept/blob/c4363c86e1e77325645d44bc8052b338031f1b30/pointcept/utils/optimizer.py#L20
        def _param_groups(cfg, model, param_dicts=None):
            if param_dicts is None:
                cfg["params"] = model.parameters()
            else:
                cfg["params"] = [dict(names=[], params=[], lr=cfg["lr"])]
                for i in range(len(param_dicts)):
                    param_group = dict(names=[], params=[], weight_decay=[])
                    if "lr" in param_dicts[i].keys():
                        param_group["lr"] = param_dicts[i]["lr"]
                    if "momentum" in param_dicts[i].keys():
                        param_group["momentum"] = param_dicts[i]["momentum"]
                    if "weight_decay" in param_dicts[i].keys():
                        param_group["weight_decay"] = param_dicts[i]["weight_decay"]
                    cfg["params"].append(param_group)

                for n, p in model.named_parameters():
                    flag = False
                    for i in range(len(param_dicts)):
                        if param_dicts[i]["match_keyword"](n):
                            cfg["params"][i + 1]["names"].append(n)
                            cfg["params"][i + 1]["params"].append(p)
                            flag = True
                            break
                    if not flag:
                        cfg["params"][0]["names"].append(n)
                        cfg["params"][0]["params"].append(p)

                for i in range(len(cfg["params"])):
                    param_names = cfg["params"][i].pop("names")
                    message = ""
                    for key in cfg["params"][i].keys():
                        if key != "params":
                            message += f" {key}: {cfg['params'][i][key]};"
                    if self.global_rank == 0:
                        self.console_logger.info(
                            f"\t[[Params Group {i + 1}]] -{message} Params: {param_names}."
                        )
            return cfg

        differential_lr = self.optimizer_cfg.pop("differential_lr", False)
        differential_lr_decay = self.optimizer_cfg.pop("differential_lr_decay", 0.85)

        if differential_lr:
            layers = [
                "embedding",
                "enc0",
                "enc1",
                "enc2",
                "enc3",
                "enc4",
                "dec3",
                "dec2",
                "dec1",
                "dec0",
            ]  # order matters!
            num_layers = len(layers)
            param_dicts = []
            for layer_id, layer in enumerate(layers):
                lr = self.optimizer_cfg["lr"] * differential_lr_decay ** (
                    num_layers - layer_id
                )
                param_dicts.append(
                    dict(
                        match_keyword=(
                            lambda n, le=layer, bl="block": le in n and bl not in n
                        ),
                        lr=lr,
                        weight_decay=(
                            0.1
                            if "enc" in layer
                            else self.optimizer_cfg["weight_decay"]
                        ),
                    )
                )

                param_dicts.append(
                    dict(
                        match_keyword=(
                            lambda n, le=layer, bl="block": le in n and bl in n
                        ),
                        lr=lr / 10.0,
                        weight_decay=(
                            0.1
                            if "enc" in layer
                            else self.optimizer_cfg["weight_decay"]
                        ),
                    )
                )
        else:
            param_dicts = [
                dict(
                    match_keyword=(lambda n: "block" in n),
                    lr=self.optimizer_cfg["lr"] / 10.0,
                    weight_decay=self.optimizer_cfg["weight_decay"],
                )
            ]

        optimizer = torch.optim.AdamW(
            **_param_groups(
                self.optimizer_cfg,
                self,
                param_dicts,
            )
        )

        self.lr_scheduler_cfg["max_lr"] = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]
        self.lr_scheduler_cfg["total_steps"] = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, **self.lr_scheduler_cfg
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            ],
        )

    def forward(self, coords, feat):
        batch_idx, coords = torch.split(coords, [1, 3], dim=1)
        pt_batch = {
            "feat": feat[:, 1:],
            "coord": coords,
            "grid_size": self.grid_size,
            "batch": batch_idx.squeeze(dim=1).long(),
        }
        out = self.backbone(pt_batch)
        out = self.head(out["feat"])
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch["coord"], batch["feat"])
        target = batch["normal"][:, 1:]

        losses = self._get_losses(out, target, batch)

        total_loss = sum(losses.values())

        for k in losses:
            self.log(f"Train/{k}", losses[k].item())
        self.log("Train/Total_Loss", total_loss.item(), prog_bar=True)
        for gid, lr in enumerate(self.lr_schedulers().get_last_lr()):
            self.log_dict({
                f"Metadata/G{gid}_LR": lr,
            })

        return total_loss

    def _get_losses(self, out, target, batch):
        losses = {
            type(c).__name__: c(
                batch["coord"],
                batch["intensity"],
                out,
                target,
                proc_tr=(self.global_step / self.trainer.estimated_stepping_batches),
                untransform_coord=batch["untransform_coord"],
                frame_id=batch["frame_id"],
                weight=batch["weight"],
            )
            for c in self.criteria
        }
        return losses

    def compute_metrics(self, pred, gt):
        error = torch.acos(
            torch.clamp(torch.cosine_similarity(pred, gt), min=-0.999999, max=0.999999)
        )
        error = torch.rad2deg(error)
        num_points = error.shape[0]
        metrics = {
            "1_mean": torch.mean(error),
            "2_median": torch.median(error),
            "3_rmse": torch.sqrt(torch.mean(torch.square(error))),
            "4_5.0°": 100.0 * (torch.sum(error < 5) / num_points),
            "5_7.5°": 100.0 * (torch.sum(error < 7.5) / num_points),
            "6_11.25°": 100.0 * (torch.sum(error < 11.25) / num_points),
            "7_22.5°": 100.0 * (torch.sum(error < 22.5) / num_points),
            "8_30.0°": 100.0 * (torch.sum(error < 30) / num_points),
        }
        return metrics

    def test_step(self, batch, batch_idx):
        out = self(batch["coord"], batch["feat"])
        target = batch["normal"][:, 1:]
        out = F.normalize(out[:, 0:3])
        metrics = self.compute_metrics(out, target)
        for k, v in metrics.items():
            self.log(
                f"Test/{k}",
                v.item(),
                batch_size=batch["batch_size"],
                sync_dist=True,
            )
        return metrics

    def validation_step(self, batch, batch_idx):
        out = self(batch["coord"], batch["feat"])
        target = batch["normal"][:, 1:]
        losses = self._get_losses(out, target, batch)
        total_loss = sum(losses.values())
        self.log(
            "Val/Val_Total_Loss",
            total_loss.item(),
            batch_size=batch["batch_size"],
            sync_dist=True,
        )

        metrics = self.compute_metrics(out[:, 0:3], target)
        for k, v in metrics.items():
            self.log(
                f"Val/{k}",
                v.item(),
                batch_size=batch["batch_size"],
                sync_dist=True,
            )

        return {"loss": total_loss, **metrics}

    def on_validation_epoch_end(self):
        if self.plm is None or not self.plm.do_update(self.current_epoch):
            return

        if self.trainer.max_epochs - 1 == self.current_epoch:
            return

        if self.global_rank == 0:
            dataloader_iterator = tqdm(self.trainer.datamodule.predict_dataloader())
        else:
            dataloader_iterator = iter(self.trainer.datamodule.predict_dataloader())
        for batch in dataloader_iterator:
            if self.global_rank == 0:
                dataloader_iterator.set_description("Generating pseudo labels")

            batch = self.trainer.datamodule.transfer_batch_to_device(
                batch, self.device, dataloader_idx=-1
            )

            pred = self(batch["coord"], batch["feat"])
            pred = F.normalize(pred[:, 0:3])

            batch_idx = batch["coord"][:, 0:1]
            batch_size_per_gpu = int(batch_idx.max().item() + 1)
            batch_idx = batch_idx + self.global_rank * batch_size_per_gpu

            batch_idx_preds = torch.cat([batch_idx, pred], dim=1).contiguous()
            batch_idx_preds = gather_all_tensors(batch_idx_preds)
            batch_idx_preds = torch.cat(batch_idx_preds, dim=0)

            frame_id = batch["frame_id"].tolist()
            all_frame_ids = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_frame_ids, frame_id)
            all_frame_ids = sum(all_frame_ids, [])

            if self.local_rank == 0:  # update PLs only once per node
                self.plm.update(
                    batch_idx_preds[:, 1:], batch_idx_preds[:, 0], all_frame_ids
                )

            self.trainer.strategy.barrier()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch["coord"], batch["feat"])
        pred = F.normalize(pred[:, 0:3], p=2, dim=1)
        gt = batch["normal"][:, 1:]

        diff = torch.acos(
            torch.clamp(torch.cosine_similarity(pred, gt), min=-0.999, max=0.999)
        )
        diff_deg = torch.rad2deg(diff)

        import open3d as o3d
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("turbo")
        diff_deg = diff_deg - diff_deg.min()
        diff_deg /= diff_deg.max()
        color = cmap(diff_deg.detach().cpu().numpy())[:, :3]

        points = batch["coord"][:, 1:].detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(color)
        pcl.normals = o3d.utility.Vector3dVector(pred)
        o3d.visualization.draw_geometries([pcl])
