import numpy as np
from scipy.spatial import KDTree


def bifurcation(G, freespace_mask, voxel_coords, voxel_resolution):
    """
    1. G에서 Ending Points(Out-degree = 0인 노드) 찾기.
    2. Ending Points 중 가장 index가 작은 노드를 선택.
    3. KD-Tree를 사용하여 selected_ending_point에 가까운 voxel을 찾고 무게중심(centroid) 계산.
    4. selected_ending_point, previous_node, centroid를 포함하는 평면을 생성하여 voxel을 분할.
    5. 두 그룹의 무게중심(centroid) 계산.
    6. 기존 edge의 실제 길이를 계산하여 0.5배 만큼 떨어진 위치에 새로운 노드 위치 계산.
    7. new_node_1, new_node_2를 포함하는 voxel의 중심으로 변환.
    8. G는 변경하지 않고, selected_ending_point, previous_ending_point, new_node_1, new_node_2 좌표 반환.

    :param G: 네트워크 그래프
    :param freespace_mask: 자유 공간 마스크 (1: 자유 공간, 0: 장애물)
    :param voxel_coords: (N, 3) 형태의 voxel 중심 좌표 배열
    :param voxel_resolution: voxel 크기 (ex: 0.1)
    :return: (selected_ending_point 좌표, previous_ending_point 좌표, new_node_1 voxel 중심 좌표, new_node_2 voxel 중심 좌표)
    """

    # Ending Points 찾기 (Out-degree = 0인 노드)
    ending_points = sorted(
        [node for node in G.nodes if G.out_degree(node) == 0 and node != 0]
    )

    if not ending_points:
        return None, None, None, None  # Ending Points가 없으면 반환

    # 가장 index가 작은 ending point 선택
    selected_ending_point = ending_points[0]  # 가장 index가 작은 노드 선택

    # freespace 좌표 추출
    free_voxel_coords = voxel_coords[freespace_mask]

    # KD-Tree 생성 (Ending Points를 트리에 저장)
    ending_positions = np.array([G.nodes[ep]["pos"] for ep in ending_points])
    kdtree = KDTree(ending_positions)

    # 각 자유 voxel에 대해 가장 가까운 Ending Point 찾기
    _, nearest_ending_index = kdtree.query(free_voxel_coords)

    # selected_ending_point에 가장 가까운 voxel만 선택
    selected_mask = nearest_ending_index == ending_points.index(selected_ending_point)
    selected_voxels = free_voxel_coords[selected_mask]

    if len(selected_voxels) == 0:
        return None, None, None, None  # 선택된 voxel이 없으면 반환

    # 선택된 voxel들의 무게중심(centroid) 계산
    centroid = np.mean(selected_voxels, axis=0)

    # selected_ending_point에 연결된 이전 노드 찾기
    previous_node = next(G.predecessors(selected_ending_point), None)
    if previous_node is None:
        return None, None, None, None  # 이전 노드가 없으면 반환

    previous_node_pos = np.array(G.nodes[previous_node]["pos"])
    selected_point_pos = np.array(G.nodes[selected_ending_point]["pos"])

    # 평면의 법선 벡터 계산 (previous_node, selected_point, centroid 포함)
    v1 = selected_point_pos - previous_node_pos
    v2 = centroid - previous_node_pos
    normal_vector = np.cross(v1, v2)  # 두 벡터의 외적을 사용해 법선 벡터 계산
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 단위 벡터화

    # voxel을 평면 기준으로 두 그룹으로 분할
    distances_to_plane = np.dot(selected_voxels - centroid, normal_vector)

    group1_voxels = selected_voxels[distances_to_plane >= 0]
    group2_voxels = selected_voxels[distances_to_plane < 0]

    # 각 그룹의 무게중심 계산
    group1_centroid = np.mean(group1_voxels, axis=0) if len(group1_voxels) > 0 else None
    group2_centroid = np.mean(group2_voxels, axis=0) if len(group2_voxels) > 0 else None

    if group1_centroid is None or group2_centroid is None:
        return None, None, None, None

    # 기존 edge의 실제 길이 계산
    edge_length = np.linalg.norm(selected_point_pos - previous_node_pos)
    move_distance = 0.4 * edge_length  # 기존 edge 길이의 0.4배

    # 새로운 노드의 좌표 계산
    new_node_1_pos = (
        selected_point_pos
        + (group1_centroid - selected_point_pos)
        / np.linalg.norm(group1_centroid - selected_point_pos)
        * move_distance
    )
    new_node_2_pos = (
        selected_point_pos
        + (group2_centroid - selected_point_pos)
        / np.linalg.norm(group2_centroid - selected_point_pos)
        * move_distance
    )

    # 새로운 노드를 포함하는 voxel 중심 좌표로 변환 (중심 좌표 기준 반올림)
    new_node_1_pos = (
        np.round(new_node_1_pos / voxel_resolution - 0.5) + 0.5
    ) * voxel_resolution
    new_node_2_pos = (
        np.round(new_node_2_pos / voxel_resolution - 0.5) + 0.5
    ) * voxel_resolution

    # G를 수정하지 않고, 필요한 좌표만 반환
    return selected_point_pos, previous_node_pos, new_node_1_pos, new_node_2_pos
