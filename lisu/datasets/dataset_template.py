import copy
from pathlib import Path
import logging
from abc import ABC, abstractmethod

import torch.utils.data as torch_data
import numpy as np
from einops import repeat
from pykdtree.kdtree import KDTree

from .dataset_utils import sample_points
from .transform import Compose


class TemplateDataset(torch_data.Dataset, ABC):
    def __init__(
        self,
        root_path: Path,
        info_paths: list[Path],
        split: str,
        limit_sequences: float,
        transform: list[dict] | None = None,
        shift_coord: list[float] = [0, 0, 0],
        pseudo_label_path: Path | None = None,
        normal_estimate_kwargs: dict = dict(k=16, sensor_origin=[0, 0, 2.5], gamma=0.2),
        use_pseudo_normals: bool = False,
        sample_points: int = -1,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.info_paths = info_paths
        self.limit_sequences = limit_sequences
        self.shift_coord = np.array(shift_coord)
        self.normal_estimate_kwargs = normal_estimate_kwargs
        self.split = split
        self.use_pseudo_normals = use_pseudo_normals
        self.sample_points = sample_points
        self.logger = logging.getLogger("lightning.pytorch.core")

        self.pseudo_label_path = pseudo_label_path
        if self.pseudo_label_path is not None and self.pseudo_label_path.is_dir():
            self.logger.info(f"The normals cache file at {self.pseudo_label_path}.")

        self.transform = Compose(transform)

        if (h2d_path := Path("h2d.npz")).is_file():
            self.logger.info(f"Using hist2d weighting from {h2d_path}")
            with np.load(h2d_path) as data:
                H, xedges, yedges = data["H"], data["xedges"], data["yedges"]
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
            self.hist2d = dict(
                H=H.flatten(),
                kde=KDTree(np.stack([X.flatten(), Y.flatten()], axis=1)),
            )
        else:
            self.hist2d = None

    @property
    @abstractmethod
    def infos(self):
        pass

    @abstractmethod
    def get_frame_id(self, info) -> str:
        pass

    @abstractmethod
    def get_lidar(self, info):
        pass

    @abstractmethod
    def get_transform(self, info) -> np.ndarray:
        pass

    def __len__(self):
        return len(self.infos)

    def compute_weights(self, normals):
        if self.hist2d is None:
            return np.ones_like(normals[:, 0])

        kde = self.hist2d["kde"]
        H = self.hist2d["H"]

        polar = np.stack(
            [
                np.arctan2(normals[:, 1], normals[:, 0]),
                np.arccos(normals[:, 2], np.linalg.norm(normals, axis=1)),
            ],
            axis=1,
        )

        k = 9
        _, idx = kde.query(polar, k=k)
        interp = H[idx].mean(axis=-1)
        weight = 1.0 - np.tanh(interp)
        return weight

    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar(info)

        if points is None:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        normal = None
        if isinstance(points, tuple):
            points, normal = points

        if self.use_pseudo_normals:
            normal_file_path = self.pseudo_label_path / f"{self.get_frame_id(info)}.npy"
            normal = np.load(normal_file_path)
            assert normal.shape[0] == points.shape[0], (
                "Something went wrong with PL generation"
            )

        data_dict = {
            "coord": points[:, 0:3],
            "intensity": points[:, 3],
            "normal": normal,
            "frame_id": self.get_frame_id(info),
            "untransform_coord": repeat(
                self.get_transform(info), "i j -> n i j", n=points.shape[0]
            ),
        }

        if normal is not None:
            normal = np.clip(normal, a_min=-1.0, a_max=1.0)
            weight = self.compute_weights(normal)
            data_dict["weight"] = weight
        else:
            data_dict.pop("normal")

        data_dict = (
            self.transform(data_dict) if self.transform is not None else data_dict
        )

        data_dict.update({"feat": data_dict["coord"]})

        if self.sample_points > 0:
            data_dict = sample_points(data_dict, self.sample_points)

        return data_dict
