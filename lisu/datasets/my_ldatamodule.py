from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.utils.data as torch_data
import lightning as L
from ignite.distributed.auto import DistributedProxySampler


def collate_batch(batch_list):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        if key in ["coord", "normal", "feat"]:
            coors = []
            if isinstance(val[0], list):
                val = [i for item in val for i in item]
            for i, coor in enumerate(val):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key in [
            "intensity",
            "mask",
            "untransform_coord",
            "normal_pred",
            "weight",
        ]:
            ret[key] = np.concatenate(val, axis=0)
        else:
            ret[key] = np.stack(val, axis=0)

    ret["batch_size"] = batch_size
    return ret


class CustomWeightedRandomSampler(torch_data.WeightedRandomSampler):
    def __init__(
        self, weights, num_samples, replacement, batch_size, frame_ids, generator=None
    ):
        super().__init__(weights, num_samples, replacement, generator)
        self.batch_size = batch_size
        self.frame_ids = frame_ids

    def __iter__(self):
        sampled_indexes = np.arange(
            0,
            self.num_samples if self.num_samples % 2 == 0 else self.num_samples - 1,
            2,
        )
        sampled_weights = self.weights[sampled_indexes]

        assert len(sampled_indexes) == sampled_weights.shape[0]

        weighted_sample_indexes = torch.multinomial(
            sampled_weights,
            len(sampled_indexes),
            self.replacement,
            generator=self.generator,
        ).tolist()

        sampled_indexes = [sampled_indexes[i] for i in weighted_sample_indexes]

        all_indexes = []
        for si in sampled_indexes:
            all_indexes += [si, si + 1]
        yield from iter(all_indexes)


class MyDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: list[torch_data.Dataset],
        batch_size: int,
        num_workers: int,
        val_dataset: torch_data.Dataset | None = None,
        test_dataset: torch_data.Dataset | None = None,
        predict_dataset: torch_data.Dataset | None = None,
        temporal: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.temporal = temporal

    def setup(self, stage: str):
        self.dataset = torch_data.ConcatDataset(self.datasets)

    def train_dataloader(self):
        kwargs = {
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "pin_memory": True,
            "num_workers": self.num_workers,
            "collate_fn": collate_batch,
            "drop_last": False,
            "timeout": 0,
            "shuffle": True,
        }

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 8

        if self.temporal:
            assert self.batch_size % 2 == 0, "Requires even batch size"

            # ensure all datasets are equally sampled
            cum_sizes = self.dataset.cummulative_sizes
            weights = []
            for i in range(len(cum_sizes)):
                if i == 0:
                    curr_ds_size = cum_sizes[i]
                else:
                    curr_ds_size = cum_sizes[i] - cum_sizes[i - 1]
                weights += [1.0 / curr_ds_size] * curr_ds_size

            fids = [
                [ds.get_frame_id(i) for i in ds.infos] for ds in self.dataset.datasets
            ]
            fids = list(itertools.chain(*fids))

            kwargs["batch_sampler"] = DistributedProxySampler(
                torch_data.BatchSampler(
                    CustomWeightedRandomSampler(
                        weights,
                        self.dataset.__len__(),
                        replacement=True,
                        batch_size=self.batch_size,
                        frame_ids=fids,
                    ),
                    batch_size=self.batch_size,
                    drop_last=False,
                ),
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
            )
            kwargs.pop("shuffle", None)
            kwargs.pop("batch_size", None)
            kwargs.pop("sampler", None)
            kwargs.pop("drop_last", None)
        else:
            cum_sizes = self.dataset.cummulative_sizes
            weights = []
            for i in range(len(cum_sizes)):
                if i == 0:
                    curr_ds_size = cum_sizes[i]
                else:
                    curr_ds_size = cum_sizes[i] - cum_sizes[i - 1]
                weights += [1.0 / curr_ds_size] * curr_ds_size
            kwargs["sampler"] = DistributedProxySampler(
                torch_data.WeightedRandomSampler(
                    weights,
                    self.dataset.__len__(),
                    replacement=self.dataset.cummulative_sizes.__len__()
                    > 1,  # only allow replacement if we have multiple datasets
                )
            )
            kwargs.pop("shuffle", None)

        dl = torch_data.DataLoader(**kwargs)
        return dl

    def predict_dataloader(self):
        ds = self.predict_dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=False, drop_last=False
        )
        dl = torch_data.DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            sampler=sampler,
            pin_memory=True,
        )
        return dl

    def val_dataloader(self):
        if self.temporal:
            batch_sampler = DistributedProxySampler(
                torch_data.BatchSampler(
                    torch_data.SequentialSampler(self.val_dataset),
                    batch_size=self.batch_size,
                    drop_last=False,
                ),
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
            )
            dl = torch_data.DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
            )
        else:
            ds = self.val_dataset
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=False, drop_last=False
            )
            dl = torch_data.DataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
                sampler=sampler,
                pin_memory=True,
            )
        return dl

    def test_dataloader(self):
        dl = torch_data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
            drop_last=False,
        )
        return dl

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, val in batch.items():
            if key == "camera_imgs":
                batch[key] = val.to(device)
            elif not isinstance(val, np.ndarray):
                continue
            elif key in [
                "frame_id",
            ]:
                continue
            elif key in ["mask"]:
                batch[key] = torch.from_numpy(val).bool().to(device)
            else:
                batch[key] = torch.from_numpy(val).float().to(device)
        return batch
