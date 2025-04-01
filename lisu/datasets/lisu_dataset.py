from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_template import TemplateDataset


class LiSuDataset(TemplateDataset):
    SPLITS = dict(
        minitrain=["Town01"],
        train=["Town03", "Town05", "Town07"],
        fulltrain=["Town01", "Town03", "Town05", "Town07"],
        val=["Town10HD"],
        fulltrainval=["Town01", "Town03", "Town05", "Town07", "Town10HD"],
        test=["Town02", "Town04", "Town06", "Town12"],
    )

    def __init__(
        self,
        root_path: Path,
        split: str,
        transform: list[dict] | None = None,
        shift_coord: list[float] = [0, 0, 0],
        pseudo_label_path: Path | None = None,
        normal_estimate_kwargs: dict = dict(k=16, sensor_origin=[0, 0, 2.5], gamma=0.2),
        info_paths: list[Path] = [],
        limit_sequences: float = 1.0,
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
            pseudo_label_path=pseudo_label_path,
            normal_estimate_kwargs=normal_estimate_kwargs,
            use_pseudo_normals=use_pseudo_normals,
            sample_points=sample_points,
        )

        self.carla_info = self.get_infos(LiSuDataset.SPLITS[split])

    def get_infos(self, towns):
        self.logger.info(f"Loading LiSu dataset from {self.root_path}")

        carla_info = []
        for town in towns:
            measurements = sorted([
                ms for ms in (self.root_path / town).glob("*") if ms.is_dir()
            ])
            for measurement in measurements:
                frames = sorted([
                    fs for fs in (measurement / "lidar").glob("*.parquet")
                ])
                frames = frames[: int(len(frames) * self.limit_sequences)]
                for frame in frames:
                    carla_info.append({
                        "town": town,
                        "measurement": measurement.stem,
                        "frame": frame.stem,
                    })
        self.logger.info(
            f"Total samples for {self.split} LiSu dataset: {len(carla_info)}"
        )
        return carla_info

    @property
    def infos(self):
        return self.carla_info

    def get_frame_id(self, info):
        town = info["town"]
        measurement = info["measurement"]
        frame = info["frame"]
        return f"{town}_{measurement}_{frame}"

    def get_lidar(self, info):
        town = info["town"]
        measurement = info["measurement"]
        frame = info["frame"]
        frame_path = self.root_path / town / measurement / "lidar" / f"{frame}.parquet"

        if not frame_path.is_file():
            self.logger.warn(
                f"Frame {self.get_frame_id(info)} cannot be find on disk - skipping it."
            )
            return None

        df = pd.read_parquet(frame_path)
        points = np.stack([df["x"], df["y"], df["z"], df["intensity"]], axis=1)
        points[:, 0:3] += self.shift_coord[np.newaxis]

        normals = np.stack(
            [df["surf_norm_x"], df["surf_norm_y"], df["surf_norm_z"]], axis=1
        )

        return points if self.use_pseudo_normals else (points, normals)

    def get_transform(self, info):
        town = info["town"]
        measurement = info["measurement"]
        frame = info["frame"]
        pose_path = self.root_path / town / measurement / "pose" / f"{frame}.txt"

        if not pose_path.is_file():
            self.logger.warn(
                f"Pose {self.get_frame_id(info)} cannot be find on disk - skipping it."
            )
            return None

        pose = np.loadtxt(pose_path)
        if not pose.shape[0] != 4 and pose.shape[1] != 4:
            self.logger.warn(f"Shape {pose.shape} is not of the expected (4, 4) shape")
            return None
        return pose
