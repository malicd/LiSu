import numpy as np
from pykdtree.kdtree import KDTree
from opt_einsum import contract
from einops import reduce


def check_symmetric(a):
    return np.allclose(a, np.transpose(a, (0, 2, 1)), atol=1e-3, rtol=0)


def estimate_surface_normals(
    pcl,
    k=9,
    sensor_origin=np.array([0, 0, 2.0]),
    gamma=0.2,
):
    # pcl: (N, 4) (x, y, z, i)
    assert pcl.ndim == 2 and pcl.shape[1] == 4

    if k >= pcl.shape[0] or np.any(
        np.isnan(pcl)
    ):  # sanity checks to avoid crashes during training
        return np.zeros_like(pcl[:, 0:3])

    kdt = KDTree(pcl[:, 0:3])
    d, indices = kdt.query(pcl[:, 0:3], k=k)

    pcl_knn = pcl[indices]  # (N, k, 4)

    weight = np.exp(-(((d[..., None]) / gamma) ** 2))
    weight_sum = reduce(weight, "n k 1 -> n 1 1", "sum")

    pcl_knn_xyz = pcl_knn[..., 0:3]
    pcl_knn_xyz_mean = reduce(pcl_knn_xyz, "n k c -> n 1 c", "mean")

    pcl_knn_xyz_shifted = weight * (pcl_knn_xyz - pcl_knn_xyz_mean)
    cov = contract("nka,nkb->nab", pcl_knn_xyz_shifted, pcl_knn_xyz_shifted)
    cov /= np.clip(weight_sum - 1.0, a_min=1.0, a_max=None)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    surf_norm = np.take_along_axis(
        eigenvectors, eigenvalues.argmin(axis=1, keepdims=True)[..., None], axis=2
    ).squeeze(axis=-1)

    # consistant viewpoint
    ray = sensor_origin - pcl[:, 0:3]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    cossim = contract("nc,nc->n", surf_norm, ray)
    surf_norm[cossim <= -0.05] *= -1

    surf_norm /= np.linalg.norm(surf_norm, axis=1, keepdims=True)  # make unit vector

    surf_norm = np.nan_to_num(
        surf_norm, nan=0.0, posinf=0.0, neginf=0.0
    )  # guard for numerical instabilities

    # surf_norm: Nx3
    return surf_norm


def plot_normals(pcl, surf_norm, mesh_location=np.array([0, 0, 0])):
    import open3d as o3d

    colors = surf_norm * 0.5 + 0.5
    points = pcl[:, 0:3]
    o3d_pcl1 = o3d.geometry.PointCloud()
    o3d_pcl1.points = o3d.utility.Vector3dVector(points)
    o3d_pcl1.colors = o3d.utility.Vector3dVector(colors)
    o3d_pcl1.normals = o3d.utility.Vector3dVector(surf_norm)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate(mesh_location)
    o3d.visualization.draw_geometries([o3d_pcl1, mesh])


def plot_intensity(pcl, i, mesh_location=np.array([0, 0, 0])):
    import open3d as o3d
    import matplotlib.pyplot as plt

    turbo_cmap = plt.get_cmap("turbo")
    attr_map_scaled = i - i.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]

    points = pcl[:, 0:3]
    o3d_pcl1 = o3d.geometry.PointCloud()
    o3d_pcl1.points = o3d.utility.Vector3dVector(points)
    o3d_pcl1.colors = o3d.utility.Vector3dVector(color)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().translate(mesh_location)
    o3d.visualization.draw_geometries([o3d_pcl1, mesh])


def _sample_points(points, num_points, p=None):
    if num_points < len(points):
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        if num_points > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(
                near_idxs,
                num_points - len(far_idxs_choice),
                replace=False,
                p=softmax(p[near_idxs]) if p is not None else p,
            )
            choice = (
                np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
                if len(far_idxs_choice) > 0
                else near_idxs_choice
            )
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False, p=softmax(p))
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            extra_choice = np.random.choice(
                choice,
                num_points - len(points),
                replace=(num_points - len(points) >= len(points)),
                p=softmax(p),
            )
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    return choice


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_points(data_dict, nb_points):
    keep_idx = _sample_points(data_dict["coord"], nb_points, p=data_dict["weight"])
    data_dict.update({
        "coord": data_dict["coord"][keep_idx],
        "feat": data_dict["feat"][keep_idx],
        "normal": data_dict["normal"][keep_idx],
        "intensity": data_dict["intensity"][keep_idx],
        "untransform_coord": data_dict["untransform_coord"][keep_idx],
        "weight": data_dict["weight"][keep_idx],
    })

    return data_dict
