import os
import pickle
from pathlib import Path

import numpy as np

from .dataset_template import TemplateDataset


class WaymoDataset(TemplateDataset):
    def __init__(
        self,
        root_path: Path,
        info_paths: list[Path],
        split: str,
        limit_sequences: float,
        transform: list[dict] | None = None,
        shift_coord: list[float] = [0, 0, 0],
        normal_estimate_kwargs: dict = dict(k=16, sensor_origin=[0, 0, 3.0], gamma=0.2),
        pseudo_label_path: Path | None = None,
        use_pseudo_normals: bool = False,
        sample_points: int = -1,
    ):
        super().__init__(
            root_path=root_path,
            info_paths=info_paths,
            split=split,
            limit_sequences=limit_sequences,
            transform=transform,
            shift_coord=shift_coord,
            normal_estimate_kwargs=normal_estimate_kwargs,
            use_pseudo_normals=use_pseudo_normals,
            pseudo_label_path=pseudo_label_path,
            sample_points=sample_points,
        )

        assert info_paths.__len__() == 1
        self.data_path = self.root_path / info_paths[0]

        self.split = split
        split_dir = self.root_path / "ImageSets" / (self.split + ".txt")
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.waymo_infos = []
        self.seq_name_to_infos = self.include_waymo_data()

    def include_waymo_data(self):
        self.logger.info("Loading Waymo dataset")
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ("%s.pkl" % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                infos = infos[: int(len(infos) * self.limit_sequences)]
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]["point_cloud"]["lidar_sequence"]] = infos

        self.waymo_infos.extend(waymo_infos[:])
        self.logger.info("Total skipped info %s" % num_skipped_infos)
        self.logger.info("Total samples for Waymo dataset: %d" % (len(waymo_infos)))

        return seq_name_to_infos

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ["training", "validation", "testing"]:
                if not sequence_file.exists():
                    temp_sequence_file = Path(
                        str(sequence_file).replace("segment", pre_text + "_segment")
                    )
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(
                    str(sequence_file).replace("_with_camera_labels", "")
                )
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    @property
    def infos(self):
        return self.waymo_infos

    def get_frame_id(self, info):
        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]
        return f"{sequence_name}_{sample_idx:04d}"

    def get_lidar(self, info):
        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]

        lidar_file = self.data_path / sequence_name / ("%04d.npy" % sample_idx)

        if not lidar_file.is_file():
            self.logger.warn(
                f"Frame {self.get_frame_id(info)} cannot be find on disk - skipping it."
            )
            return None

        point_features = np.load(
            lidar_file
        )  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        points_all[:, 0:3] = points_all[:, 0:3] + self.shift_coord[np.newaxis]

        if points_all.shape[0] < 10000:
            self.logger.warn(
                f"Frame {self.get_frame_id(info)} has {points_all.shape[0]} points and will be skipped."
            )
            return None

        return points_all

    def get_transform(self, info):
        return info["pose"]
