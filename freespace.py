import numpy as np

def update_freespace(start, end, diameter, voxel_coords, voxel_resolution, freespace_mask):
    """
    하나의 원통형 edge가 지나가는 voxel을 찾아 해당 voxel의 freespace_mask 값을 0(False)로 설정.
    
    :param start: (x, y, z) - 원통형 edge의 시작점 좌표
    :param end: (x, y, z) - 원통형 edge의 끝점 좌표
    :param diameter: float - 원통의 지름 (edge의 두께)
    :param voxel_coords: numpy array (N, 3) - 전체 voxel 중심 좌표 배열
    :param voxel_resolution: float - voxel 크기 (ex: 0.1)
    :param freespace_mask: numpy array (N,) - 자유 공간을 나타내는 마스크 (True: 자유, False: 장애물)
    :return: 업데이트된 freespace_mask
    """

    # 1. 시작점과 끝점을 numpy 배열로 변환
    start = np.array(start)
    end = np.array(end)

    # 2. edge를 따라가는 직선 경로 샘플링
    num_samples = int(np.linalg.norm(end - start) / voxel_resolution) + 1
    sampled_points = np.linspace(start, end, num_samples)

    # 3. diameter를 고려한 voxel 제거
    radius = diameter / 2  # 직경 → 반지름 변환
    for point in sampled_points:
        # voxel_coords 중에서 radius 이내에 있는 좌표 찾기
        distances = np.linalg.norm(voxel_coords - point, axis=1)
        inside_voxel = distances < radius
        freespace_mask[inside_voxel] = False  # 해당 voxel을 장애물로 설정
    
    return freespace_mask
