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
    space_size,
    voxel_resolution,
):
    selected_point_pos = np.asarray(
        selected_point_pos,
        dtype=float,
    )
    space_size = np.asarray(
        space_size,
        dtype=float,
    )

    # 각 축에서 가장 바깥쪽 voxel의 중심 좌표
    lower_boundary = np.full(
        3,
        voxel_resolution / 2,
    )
    upper_boundary = space_size - voxel_resolution / 2

    tolerance = voxel_resolution * 1e-6

    at_lower_boundary = np.isclose(
        selected_point_pos,
        lower_boundary,
        atol=tolerance,
        rtol=0.0,
    )
    at_upper_boundary = np.isclose(
        selected_point_pos,
        upper_boundary,
        atol=tolerance,
        rtol=0.0,
    )

    # x, y, z 중 하나라도 최외곽 voxel이면 True
    return bool(np.any(at_lower_boundary | at_upper_boundary))
