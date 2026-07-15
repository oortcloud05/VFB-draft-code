import numpy as np
from scipy.spatial import KDTree
from limit_angle import limit_angle
from terminal_condition import branch_length_valid
from terminal_condition import branch_diameter_valid


def bifurcation(G, freespace_mask, voxel_coords, voxel_resolution):
    """
    1. G에서 Ending Points(Out-degree = 0인 노드) 찾기.
    2. Ending Points 중 가장 index가 작은 노드를 선택.
    3. KD-Tree를 사용하여 selected_ending_point에 가까운 voxel을 찾고 무게중심(centroid) 계산.
    4. selected_ending_point, previous_node, centroid를 포함하는 평면을 생성하여 voxel을 분할.
    5. 두 그룹의 무게중심(centroid) 계산.
    6. 기존 edge의 실제 길이를 계산하여 0.4배 만큼 떨어진 위치에 새로운 노드 위치 계산.
    7. new_node_1, new_node_2를 포함하는 voxel의 중심으로 변환.
    8. G는 변경하지 않고, selected_ending_point, previous_ending_point, new_node_1, new_node_2 좌표 반환.

    위 과정이 실패할 경우, 다음 index의 노드에서 반복

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

    # freespace 좌표 추출
    free_voxel_coords = voxel_coords[freespace_mask]

    # freespace에 남은 voxel이 없는 경우, branching 종료
    if len(free_voxel_coords) == 0:
        return None, None, None, None

    # KD-Tree 생성 (Ending Points를 트리에 저장)
    ending_positions = np.array([G.nodes[ep]["pos"] for ep in ending_points])
    kdtree = KDTree(ending_positions)

    # 각 자유 voxel에 대해 가장 가까운 Ending Point 찾기
    _, nearest_ending_index = kdtree.query(free_voxel_coords)

    # 각 Ending Point를 순서대로 시도
    for selected_index, selected_ending_point in enumerate(ending_points):
        # selected_ending_point에 연결된 이전 노드 찾기
        previous_node = next(G.predecessors(selected_ending_point), None)

        if previous_node is None:
            continue

        # terminal condition 2: 가지 직경 검사
        parent_diameter = G.edges[
            previous_node,
            selected_ending_point,
        ].get("diameter", 1.0)
        new_diameter = parent_diameter * 0.8
        if not branch_diameter_valid(new_diameter):
            continue

        # 현재 Ending Point에 배정된 voxel 선택
        selected_mask = nearest_ending_index == selected_index
        selected_voxels = free_voxel_coords[selected_mask]

        # 배정된 voxel이 없으면 다음 Ending Point 시도
        if len(selected_voxels) == 0:
            continue

        previous_node_pos = np.array(G.nodes[previous_node]["pos"])
        selected_point_pos = np.array(G.nodes[selected_ending_point]["pos"])

        # 선택된 voxel들의 무게중심(centroid) 계산
        centroid = np.mean(selected_voxels, axis=0)

        # 평면의 법선 벡터 계산 (previous_node, selected_point, centroid 포함)
        v1 = selected_point_pos - previous_node_pos
        v2 = centroid - previous_node_pos
        normal_vector = np.cross(v1, v2)  # 두 벡터의 외적을 사용해 법선 벡터 계산
        normal_length = np.linalg.norm(normal_vector)

        # 평면을 만들 수 없으면 다음 Ending Point 시도
        if np.isclose(normal_length, 0.0):
            continue

        normal_vector = normal_vector / normal_length  # 단위 벡터화

        # voxel을 평면 기준으로 두 그룹으로 분할
        distances_to_plane = np.dot(selected_voxels - centroid, normal_vector)

        group1_voxels = selected_voxels[distances_to_plane >= 0]
        group2_voxels = selected_voxels[distances_to_plane < 0]

        # 어느 한쪽 그룹이 비었으면 다음 Ending Point 시도
        if len(group1_voxels) == 0 or len(group2_voxels) == 0:
            continue

        # 각 그룹의 무게중심 계산
        group1_centroid = np.mean(group1_voxels, axis=0)
        group2_centroid = np.mean(group2_voxels, axis=0)

        # 자녀 방향을 계산할 수 없으면 다음 Ending Point 시도
        if np.isclose(
            np.linalg.norm(group1_centroid - selected_point_pos), 0.0
        ) or np.isclose(np.linalg.norm(group2_centroid - selected_point_pos), 0.0):
            continue

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

        # 최대각(60도) 보정
        new_node_1_pos = limit_angle(
            previous_node_pos,
            selected_point_pos,
            new_node_1_pos,
            max_angle_deg=60.0,
        )
        new_node_2_pos = limit_angle(
            previous_node_pos,
            selected_point_pos,
            new_node_2_pos,
            max_angle_deg=60.0,
        )

        # 새로운 노드를 포함하는 voxel 중심 좌표로 변환 (중심 좌표 기준 반올림)
        new_node_1_pos = (
            np.round(new_node_1_pos / voxel_resolution - 0.5) + 0.5
        ) * voxel_resolution
        new_node_2_pos = (
            np.round(new_node_2_pos / voxel_resolution - 0.5) + 0.5
        ) * voxel_resolution

        # terminal condition 1: 가지 길이
        if not branch_length_valid(
            selected_point_pos,
            new_node_1_pos,
        ):
            continue
        if not branch_length_valid(
            selected_point_pos,
            new_node_2_pos,
        ):
            continue

        # G를 수정하지 않고, 필요한 좌표만 반환
        return selected_point_pos, previous_node_pos, new_node_1_pos, new_node_2_pos

    # 모든 Ending Point의 분할이 실패한 경우에만 종료
    return None, None, None, None
