import numpy as np


## edge 주변의 voxel만 탐색하도록 하여 계산량 줄임


def update_freespace(
    start, end, diameter, voxel_coords, voxel_resolution, freespace_mask
):
    start = np.array(start)
    end = np.array(end)
    radius = diameter / 2

    lower = np.minimum(start, end) - radius
    upper = np.maximum(start, end) + radius

    candidate_mask = (
        freespace_mask
        & (voxel_coords[:, 0] >= lower[0])
        & (voxel_coords[:, 0] <= upper[0])
        & (voxel_coords[:, 1] >= lower[1])
        & (voxel_coords[:, 1] <= upper[1])
        & (voxel_coords[:, 2] >= lower[2])
        & (voxel_coords[:, 2] <= upper[2])
    )

    candidate_indices = np.where(candidate_mask)[0]
    candidate_voxels = voxel_coords[candidate_indices]

    edge_vec = end - start
    edge_len_sq = np.dot(edge_vec, edge_vec)

    if edge_len_sq == 0:
        distances = np.linalg.norm(candidate_voxels - start, axis=1)
    else:
        voxel_vec = candidate_voxels - start
        t = np.dot(voxel_vec, edge_vec) / edge_len_sq
        t = np.clip(t, 0, 1)
        closest_points = start + t[:, None] * edge_vec
        distances = np.linalg.norm(candidate_voxels - closest_points, axis=1)

    inside = distances < radius
    freespace_mask[candidate_indices[inside]] = False

    return freespace_mask
