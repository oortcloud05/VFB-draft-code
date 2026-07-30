import numpy as np


# 1. 가지 길이 제한
def branch_length_valid(
    selected_point_pos,
    new_node_pos,
    min_branch_length=0.1,
):
    branch_length = np.linalg.norm(new_node_pos - selected_point_pos)

    return branch_length >= min_branch_length


# 2. 가지 직경 제한
def branch_diameter_valid(
    branch_diameter,
    min_branch_diameter=0.025,
):
    return branch_diameter >= min_branch_diameter


# 3. 전체 voxel 공간의 boundary 도달 여부
def ending_point_at_boundary(
    selected_point_pos,
    organ_mask_3d,
    voxel_resolution,
):

    selected_point_pos = np.asarray(
        selected_point_pos,
        dtype=float,
    )

    grid_shape = np.asarray(
        organ_mask_3d.shape,
        dtype=int,
    )

    voxel_index = np.rint(selected_point_pos / voxel_resolution - 0.5).astype(int)

    # 노드 자체가 전체 voxel 공간 밖에 있는 경우
    if np.any(voxel_index < 0) or np.any(voxel_index >= grid_shape):
        return True

    x_index, y_index, z_index = voxel_index

    # 노드 자체가 장기 외부인 경우
    if not organ_mask_3d[x_index, y_index, z_index]:
        return True

    neighbor_offsets = np.array(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=int,
    )

    for offset in neighbor_offsets:
        neighbor_index = voxel_index + offset

        # 인접 voxel이 전체 정육면체 밖에 있는 경우
        if np.any(neighbor_index < 0) or np.any(neighbor_index >= grid_shape):
            return True

        neighbor_x, neighbor_y, neighbor_z = neighbor_index

        # 인접 voxel이 장기 외부인 경우
        if not organ_mask_3d[
            neighbor_x,
            neighbor_y,
            neighbor_z,
        ]:
            return True

    return False
