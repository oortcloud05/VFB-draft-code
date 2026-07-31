import numpy as np


# 1. 여러 voxel 중심점들이 start->end 벡터와 떨어진 거리 계산
def distance_to_segment(points, start, end):

    edge_vec = end - start
    edge_len_sq = np.dot(edge_vec, edge_vec)

    # 시작점이 끝점과 같은 경우, 시작점과 point 사이의 거리 반환
    if edge_len_sq == 0:
        return np.linalg.norm(points - start, axis=1)

    # 각 point(voxel 중심점)를 edge 벡터 위에 투영해 거리 계산, 반환
    point_vec = points - start
    t = np.dot(point_vec, edge_vec) / edge_len_sq
    t = np.clip(t, 0, 1)
    closest_points = start + t[:, None] * edge_vec
    return np.linalg.norm(points - closest_points, axis=1)


# 2. start->end 가지가 기존 occupied voxel과 겹치는지 확인
def candidate_edge_overlaps_existing(
    start,
    end,
    diameter,
    voxel_coords,
    freespace_mask,
):

    radius = diameter / 2

    # 벡터 bounding box (start->end 벡터가 공간대각선 + radius만큼 더한 직육면체)
    lower = np.minimum(start, end) - radius
    upper = np.maximum(start, end) + radius

    # bounding box 내부 occupied voxel만 골라냄
    occupied_mask = ~freespace_mask
    bbox_mask = (
        occupied_mask
        & (voxel_coords[:, 0] >= lower[0])
        & (voxel_coords[:, 0] <= upper[0])
        & (voxel_coords[:, 1] >= lower[1])
        & (voxel_coords[:, 1] <= upper[1])
        & (voxel_coords[:, 2] >= lower[2])
        & (voxel_coords[:, 2] <= upper[2])
    )
    occupied_candidates = voxel_coords[bbox_mask]

    # 새 branch는 selected point에서 시작하므로,
    # 시작점 주변의 기존 parent branch voxel은 overlap 판정에서 제외하도록 update
    distance_from_start = np.linalg.norm(occupied_candidates - start, axis=1)
    occupied_candidates = occupied_candidates[distance_from_start > diameter]

    # 검사할 occupied voxel이 없는 경우, 겹침 없음
    if len(occupied_candidates) == 0:
        return False

    # occupied voxel에 대해 start->end 벡터와 거리 계산
    # radius 보다 가까운 경우, 겹침 존재
    distances = distance_to_segment(occupied_candidates, start, end)
    return np.any(distances < radius)


# 3. raw new node ~ selected point 사이 free voxel에 대해 scoring, 최적 후보 반환
def avoid_intersection(
    selected_point_pos,
    raw_new_node_pos,
    diameter,
    voxel_coords,
    voxel_resolution,
    freespace_mask,
    min_length_ratio=0.3,  # 원래 가지 길이에 비해 얼마나 짧은 후보까지 허용할지
    search_radius_ratio=1.0,  # raw_new_node_pos 주변 voxel을 얼마나 넓게 포함할지
    angle_weight=1.0,  # 각도 차이 score 가중치
    endpoint_weight=0.7,  # raw_new_node_pos와의 거리 score 가중치
    length_weight=0.5,  # 원래 길이와의 차이 score 가중치
):

    selected_point_pos = np.array(selected_point_pos)
    raw_new_node_pos = np.array(raw_new_node_pos)

    # 최초로 계산되는 branch 벡터의 길이
    direction = raw_new_node_pos - selected_point_pos
    original_length = np.linalg.norm(direction)

    # 길이가 0인 경우, 새 branch를 만들 수 없음
    if original_length == 0:
        return None

    # 원래 centroid 방향의 단위 벡터
    direction_unit = direction / original_length

    free_voxels = voxel_coords[freespace_mask]

    # free voxel이 없는 경우, 새 branch를 만들 수 없음
    if len(free_voxels) == 0:
        return None

    # 각 free voxel에 대한 벡터 생성, 후보 branch 길이 계산
    vectors = free_voxels - selected_point_pos
    lengths = np.linalg.norm(vectors, axis=1)

    # 벡터 길이로 voxel 후보 제한
    valid_length_mask = (lengths >= original_length * min_length_ratio) & (
        lengths <= original_length
    )

    # 기존 raw end 근처 (voxel_resolution 이내) free voxel도 후보로 포함
    endpoint_distances = np.linalg.norm(free_voxels - raw_new_node_pos, axis=1)
    nearby_raw_endpoint_mask = (
        endpoint_distances <= voxel_resolution * search_radius_ratio
    )

    # 최종 end point 후보 voxel
    nonzero_length_mask = lengths > 0
    candidate_mask = valid_length_mask & nonzero_length_mask
    candidate_mask = candidate_mask | nearby_raw_endpoint_mask

    candidate_voxels = free_voxels[candidate_mask]
    candidate_lengths = lengths[candidate_mask]
    candidate_endpoint_distances = endpoint_distances[candidate_mask]

    # 후보가 없는 경우, 새 branch를 만들 수 없음
    if len(candidate_voxels) == 0:
        return None

    # 각 후보 voxel 방향의 단위 벡터
    candidate_vectors = candidate_voxels - selected_point_pos
    candidate_unit_vectors = candidate_vectors / candidate_lengths[:, None]

    # 원래 centroid 방형과 후보 방향 사이 cosine (for 각도 scoring)
    cos_angles = np.dot(candidate_unit_vectors, direction_unit)
    cos_angles = np.clip(cos_angles, -1, 1)

    # scoring (방향 틀어짐, raw endpoint와 멀어짐, branch 길이 달라짐)
    angle_penalty = 1 - cos_angles
    endpoint_penalty = candidate_endpoint_distances / original_length
    length_penalty = np.abs(original_length - candidate_lengths) / original_length

    scores = (
        angle_weight * angle_penalty
        + endpoint_weight * endpoint_penalty
        + length_weight * length_penalty
    )

    # score 순위화 (낮을수록 좋은 후보)
    ranked_indices = np.argsort(scores)

    # score가 좋은 후보부터 endpoint로 지정해 새로운 branch 만들어서 겹침 조사
    for index in ranked_indices:
        candidate_endpoint = candidate_voxels[index]

        overlaps = candidate_edge_overlaps_existing(
            selected_point_pos,
            candidate_endpoint,
            diameter,
            voxel_coords,
            freespace_mask,
        )

        if not overlaps:
            return candidate_endpoint

    # 모든 후보가 겹침이 생기는 경우, 새 branch를 만들 수 없음
    return None
