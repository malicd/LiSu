import random

import numpy as np
from einops import einsum, repeat


class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            shift = [shift_x, shift_y, shift_z]
            data_dict["coord"] += shift

            data_dict["untransform_coord"] = self.revert(
                data_dict["untransform_coord"], shift
            )
        return data_dict

    @staticmethod
    def revert(untransform_coord, shift):
        tmp_shift = np.eye(4)
        tmp_shift[:3, -1] -= shift
        untransform_coord = einsum(untransform_coord, tmp_shift, "n i j, j k -> n i k")
        return untransform_coord


class Crop(object):
    def __init__(self, point_cloud_range=None, ego_center_radius=0):
        self.point_cloud_range = point_cloud_range
        self.ego_center_radius = ego_center_radius

    def __call__(self, data_dict):
        if self.point_cloud_range is not None:
            coord = data_dict["coord"]
            mask = (
                (coord[:, 0] >= self.point_cloud_range[0])
                & (coord[:, 0] <= self.point_cloud_range[3])
                & (coord[:, 1] >= self.point_cloud_range[1])
                & (coord[:, 1] <= self.point_cloud_range[4])
                & (coord[:, 2] >= self.point_cloud_range[2])
                & (coord[:, 2] <= self.point_cloud_range[5])
            )
            data_dict["coord"] = data_dict["coord"][mask]
            if "normal" in data_dict:
                data_dict["normal"] = data_dict["normal"][mask]
            if "intensity" in data_dict:
                data_dict["intensity"] = data_dict["intensity"][mask]
            if "untransform_coord" in data_dict:
                data_dict["untransform_coord"] = data_dict["untransform_coord"][mask]
            if "weight" in data_dict:
                data_dict["weight"] = data_dict["weight"][mask]

        if self.ego_center_radius > 0.0:
            coord = data_dict["coord"]
            mask = ~(
                (np.abs(coord[:, 0]) < self.ego_center_radius)
                & (np.abs(coord[:, 1]) < self.ego_center_radius)
            )
            data_dict["coord"] = data_dict["coord"][mask]
            if "normal" in data_dict:
                data_dict["normal"] = data_dict["normal"][mask]
            if "intensity" in data_dict:
                data_dict["intensity"] = data_dict["intensity"][mask]
            if "untransform_coord" in data_dict:
                data_dict["untransform_coord"] = data_dict["untransform_coord"][mask]
            if "weight" in data_dict:
                data_dict["weight"] = data_dict["weight"][mask]
        return data_dict


class RandomScale(object):
    def __init__(self, scale=None):
        self.scale = scale if scale is not None else [0.95, 1.05]

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(self.scale[0], self.scale[1], 1)
            data_dict["coord"] *= scale
            data_dict["untransform_coord"] = self.revert(
                data_dict["untransform_coord"], scale
            )
        return data_dict

    @staticmethod
    def revert(untransform_coord, scale):
        tmp_scale = np.eye(4)
        tmp_scale[:3, :3] /= scale
        untransform_coord = einsum(untransform_coord, tmp_scale, "n i j, j k -> n i k")
        return untransform_coord


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if flip_x := np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] *= -1.0
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] *= -1.0
        if flip_y := np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] *= -1.0
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] *= -1.0
        data_dict["untransform_coord"] = self.revert(
            data_dict["untransform_coord"], flip_x, flip_y
        )
        return data_dict

    @staticmethod
    def revert(untransform_coord, flip_x, flip_y):
        tmp_flip = np.eye(4)
        tmp_flip[0, 0] = -1 if flip_x else 1
        tmp_flip[1, 1] = -1 if flip_y else 1
        untransform_coord = einsum(untransform_coord, tmp_flip, "n i j, j k -> n i k")
        return untransform_coord


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
            data_dict["untransform_coord"] = self.revert(
                data_dict["untransform_coord"], jitter
            )

        if "normals" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["normals"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["normals"] += jitter
        return data_dict

    @staticmethod
    def revert(untransform_coord, jitter):
        tmp_jitter = np.eye(4)
        tmp_jitter = repeat(tmp_jitter, "i j -> n i j", n=untransform_coord.shape[0])
        tmp_jitter[:, :3, -1] = -1.0 * jitter
        untransform_coord = einsum(
            untransform_coord, tmp_jitter, "n i j, n j k -> n i k"
        )
        return untransform_coord


class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        if "intensity" in data_dict.keys():
            data_dict["intensity"] = data_dict["intensity"][shuffle_index]
        if "untransform_coord" in data_dict.keys():
            data_dict["untransform_coord"] = data_dict["untransform_coord"][
                shuffle_index
            ]
        if "weight" in data_dict.keys():
            data_dict["weight"] = data_dict["weight"][shuffle_index]
        return data_dict


class RandomRotate(object):
    def __init__(self, angle=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))

            data_dict["untransform_coord"] = self.revert(
                data_dict["untransform_coord"], rot_t
            )

        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

    @staticmethod
    def revert(untransform_coord, rot):
        tmp_rot = np.eye(4)
        tmp_rot[:3, :3] = rot.T
        untransform_coord = einsum(untransform_coord, tmp_rot, "n i j, j k -> n i k")
        return untransform_coord


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(globals()[t_cfg.pop("name")](**t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
