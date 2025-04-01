import torch
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('medium')

cli = LightningCLI(seed_everything_default=42, run=False)
cli.trainer.validate(cli.model, datamodule=cli.datamodule)
cli.trainer.fit(cli.model, datamodule=cli.datamodule)
