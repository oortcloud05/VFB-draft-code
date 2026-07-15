import numpy as np

from limit_angle import calculate_branch_angle
from terminal_condition import branch_length_valid


# 60도 보정 이후 raw node 주변에서 조건을 만족하는
# voxel 중심과 raw node까지의 거리 반환
def find_valid_voxel_candidates(
    adjusted_raw_node_pos,
    previous_node_pos,
    selected_point_pos,
    voxel_coords,
    freespace_mask,
    voxel_resolution,
    space_size,
    search_layers=1,
    min_branch_length=0.1,
    max_angle_deg=60.0,
):

    adjusted_raw_node_pos = np.asarray(
        adjusted_raw_node_pos,
        dtype=float,
    )
    previous_node_pos = np.asarray(
        previous_node_pos,
        dtype=float,
    )
    selected_point_pos = np.asarray(
        selected_point_pos,
        dtype=float,
    )
    space_size = np.asarray(
        space_size,
        dtype=float,
    )

    # 각 축의 voxel 개수
    grid_shape = np.rint(space_size / voxel_resolution).astype(int)

    # adjusted raw node와 가장 가까운 voxel의 index
    center_index = np.rint(adjusted_raw_node_pos / voxel_resolution - 0.5).astype(int)

    # 주변 voxel index 생성
    offset_range = np.arange(
        -search_layers,
        search_layers + 1,
    )

    offset_x, offset_y, offset_z = np.meshgrid(
        offset_range,
        offset_range,
        offset_range,
        indexing="ij",
    )

    offsets = np.column_stack(
        [
            offset_x.ravel(),
            offset_y.ravel(),
            offset_z.ravel(),
        ]
    )

    candidate_indices = center_index + offsets

    # 전체 voxel 공간 밖의 후보 제거
    inside_space = np.all(
        (candidate_indices >= 0) & (candidate_indices < grid_shape),
        axis=1,
    )

    candidate_indices = candidate_indices[inside_space]

    if len(candidate_indices) == 0:
        return np.empty((0, 3)), np.empty(0)

    # 3차원 voxel index를 voxel_coords의 1차원 index로 변환
    flat_indices = np.ravel_multi_index(
        candidate_indices.T,
        tuple(grid_shape),
    )

    # 현재 free voxel인 후보만 유지
    free_candidates = freespace_mask[flat_indices]

    flat_indices = flat_indices[free_candidates]

    if len(flat_indices) == 0:
        return np.empty((0, 3)), np.empty(0)

    candidate_positions = voxel_coords[flat_indices]

    valid_positions = []
    valid_distances = []

    for candidate_pos in candidate_positions:
        # terminal condition 1: 최종 가지 길이 검사
        if not branch_length_valid(
            selected_point_pos,
            candidate_pos,
            min_branch_length=min_branch_length,
        ):
            continue

        # 최종 가지 각도 검사
        angle_deg = calculate_branch_angle(
            previous_node_pos,
            selected_point_pos,
            candidate_pos,
        )

        if not np.isfinite(angle_deg):
            continue

        if angle_deg > max_angle_deg + 1e-8:
            continue

        # 조건을 통과한 voxel과 보정된 raw node 사이 거리
        distance_to_raw_node = np.linalg.norm(candidate_pos - adjusted_raw_node_pos)

        valid_positions.append(candidate_pos)
        valid_distances.append(distance_to_raw_node)

    if not valid_positions:
        return np.empty((0, 3)), np.empty(0)

    valid_positions = np.asarray(valid_positions)
    valid_distances = np.asarray(valid_distances)

    # 보정된 raw node와 가까운 순서로 정렬
    sorted_indices = np.argsort(valid_distances)

    return (
        valid_positions[sorted_indices],
        valid_distances[sorted_indices],
    )


# 두 branch가 같은 voxel 선택하는 문제 해결
def select_voxel_pair(
    adjusted_raw_node_1_pos,
    adjusted_raw_node_2_pos,
    previous_node_pos,
    selected_point_pos,
    voxel_coords,
    freespace_mask,
    voxel_resolution,
    space_size,
    search_layers=1,
    min_branch_length=0.1,
    max_angle_deg=60.0,
):

    candidates_1, distances_1 = find_valid_voxel_candidates(
        adjusted_raw_node_1_pos,
        previous_node_pos,
        selected_point_pos,
        voxel_coords,
        freespace_mask,
        voxel_resolution,
        space_size,
        search_layers=search_layers,
        min_branch_length=min_branch_length,
        max_angle_deg=max_angle_deg,
    )

    candidates_2, distances_2 = find_valid_voxel_candidates(
        adjusted_raw_node_2_pos,
        previous_node_pos,
        selected_point_pos,
        voxel_coords,
        freespace_mask,
        voxel_resolution,
        space_size,
        search_layers=search_layers,
        min_branch_length=min_branch_length,
        max_angle_deg=max_angle_deg,
    )

    if len(candidates_1) == 0 or len(candidates_2) == 0:
        return None

    best_pair = None
    best_total_distance = np.inf

    for index_1, candidate_1 in enumerate(candidates_1):
        for index_2, candidate_2 in enumerate(candidates_2):
            # 두 자녀 node가 같은 voxel이면 제외
            if np.allclose(
                candidate_1,
                candidate_2,
            ):
                continue

            total_distance = distances_1[index_1] + distances_2[index_2]

            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_pair = (
                    candidate_1,
                    candidate_2,
                )

    return best_pair
