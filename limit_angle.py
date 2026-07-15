import numpy as np


# 부모 branch와 자녀 branch 사이의 각도 제한
def limit_angle(
    previous_node_pos,
    selected_point_pos,
    new_node_pos,
    max_angle_deg=60.0,
):

    previous_node_pos = np.asarray(previous_node_pos, dtype=float)
    selected_point_pos = np.asarray(selected_point_pos, dtype=float)
    new_node_pos = np.asarray(new_node_pos, dtype=float)

    # 새로 만들어지는 가지의 벡터 계산
    previous_vector = selected_point_pos - previous_node_pos
    new_vector = new_node_pos - selected_point_pos

    previous_length = np.linalg.norm(previous_vector)
    new_length = np.linalg.norm(new_vector)

    if previous_length == 0 or new_length == 0:
        return new_node_pos

    previous_unit_vector = previous_vector / previous_length
    new_unit_vector = new_vector / new_length

    # cosine 값으로 가지 간 각도 역산
    cos_angle = np.clip(
        np.dot(previous_unit_vector, new_unit_vector),
        -1.0,
        1.0,
    )
    angle_deg = np.degrees(np.arccos(cos_angle))

    # 60도 이하인 경우, 변동 없이 출력
    if angle_deg <= max_angle_deg:
        return new_node_pos

    # 60도 초과인 경우, 보정이 필요함

    # 축 생성을 위해 새로운 branch에서 이전 branch와 수직인 성분 추출
    perpendicular_vector = new_unit_vector - cos_angle * previous_unit_vector
    perpendicular_length = np.linalg.norm(perpendicular_vector)

    # 두 branch가 일직선인 경우 예외 처리
    if perpendicular_length == 0:
        reference_axis = np.zeros(3)
        reference_axis[np.argmin(np.abs(previous_unit_vector))] = 1.0
        perpendicular_vector = np.cross(previous_unit_vector, reference_axis)
        perpendicular_length = np.linalg.norm(perpendicular_vector)

    # 최종 축
    perpendicular_unit_vector = perpendicular_vector / perpendicular_length
    max_angle_rad = np.radians(max_angle_deg)

    # 60도 방향으로 새로운 branch의 벡터 계산
    adjusted_new_unit_vector = (
        np.cos(max_angle_rad) * previous_unit_vector
        + np.sin(max_angle_rad) * perpendicular_unit_vector
    )

    # 최종적으로 보정된 new node의 position 출력
    return selected_point_pos + adjusted_new_unit_vector * new_length


# 부모 branch와 자녀 branch 사이 각도 계산
def calculate_branch_angle(
    previous_node_pos,
    selected_point_pos,
    new_node_pos,
):

    previous_node_pos = np.asarray(previous_node_pos, dtype=float)
    selected_point_pos = np.asarray(selected_point_pos, dtype=float)
    new_node_pos = np.asarray(new_node_pos, dtype=float)

    parent_vector = selected_point_pos - previous_node_pos
    child_vector = new_node_pos - selected_point_pos

    parent_length = np.linalg.norm(parent_vector)
    child_length = np.linalg.norm(child_vector)

    if np.isclose(parent_length, 0.0) or np.isclose(child_length, 0.0):
        return np.nan

    parent_unit_vector = parent_vector / parent_length
    child_unit_vector = child_vector / child_length

    cos_angle = np.clip(
        np.dot(parent_unit_vector, child_unit_vector),
        -1.0,
        1.0,
    )

    return np.degrees(np.arccos(cos_angle))
