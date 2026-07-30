import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from scipy.spatial import KDTree


# 덩어리 생성 함수
def generate_random_mass(
    space_size=(10.0, 10.0, 10.0),
    voxel_resolution=0.05,
    min_volume_ratio=0.15,
    max_volume_ratio=0.40,
    noise_strength=0.15,
    noise_smoothness=0.7,
    margin_voxels=2,
    max_attempts=50,
    seed=None,
):
    """
    타원체 표면에 부드러운 3차원 노이즈를 적용하여
    하나로 연결된 무작위 덩어리를 생성한다.

    True  : 덩어리 내부의 자유 공간
    False : 덩어리 외부의 차단 공간

    반환되는 1차원 mask는 다음 voxel_coords의 순서와 일치한다.

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    voxel_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    """

    space_size = np.asarray(space_size, dtype=float)

    if space_size.shape != (3,):
        raise ValueError("space_size는 (x, y, z) 형태여야 합니다.")
    if voxel_resolution <= 0:
        raise ValueError("voxel_resolution은 0보다 커야 합니다.")
    if not 0.0 < min_volume_ratio < max_volume_ratio < 1.0:
        raise ValueError(
            "부피 비율은 0 < min_volume_ratio < max_volume_ratio < 1을 만족해야 합니다."
        )
    if noise_strength < 0:
        raise ValueError("noise_strength는 0 이상이어야 합니다.")
    if noise_smoothness <= 0:
        raise ValueError("noise_smoothness는 0보다 커야 합니다.")

    grid_shape = np.rint(space_size / voxel_resolution).astype(int)
    if np.any(grid_shape <= 0):
        raise ValueError("모든 축에 voxel이 하나 이상 존재해야 합니다.")
    if margin_voxels < 0:
        raise ValueError("margin_voxels는 0 이상이어야 합니다.")
    if margin_voxels * 2 >= np.min(grid_shape):
        raise ValueError("margin_voxels가 공간 크기에 비해 너무 큽니다.")

    rng = np.random.default_rng(seed)

    # voxel 중심 좌표 계산
    x = np.arange(grid_shape[0]) * voxel_resolution + voxel_resolution / 2
    y = np.arange(grid_shape[1]) * voxel_resolution + voxel_resolution / 2
    z = np.arange(grid_shape[2]) * voxel_resolution + voxel_resolution / 2

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    six_neighbor_structure = np.zeros((3, 3, 3), dtype=np.uint8)
    six_neighbor_structure[1, 1, 1] = 1
    six_neighbor_structure[0, 1, 1] = 1
    six_neighbor_structure[2, 1, 1] = 1
    six_neighbor_structure[1, 0, 1] = 1
    six_neighbor_structure[1, 2, 1] = 1
    six_neighbor_structure[1, 1, 0] = 1
    six_neighbor_structure[1, 1, 2] = 1

    # cm 단위의 smoothness를 voxel 단위로 변환
    noise_sigma = max(
        noise_smoothness / voxel_resolution,
        0.5,
    )

    for _ in range(max_attempts):
        # 덩어리 중심을 공간 중앙 부근에서 무작위로 이동
        center_shift = (
            rng.uniform(
                low=-0.05,
                high=0.05,
                size=3,
            )
            * space_size
        )

        center = space_size / 2 + center_shift

        # 각 축 반지름을 무작위로 설정
        radius_fractions = rng.uniform(
            low=0.34,
            high=0.46,
            size=3,
        )
        radii = space_size * radius_fractions

        # 무작위 3차원 회전 행렬 생성
        random_matrix = rng.normal(size=(3, 3))
        rotation_matrix, _ = np.linalg.qr(random_matrix)

        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1

        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]

        # 좌표를 회전된 타원체 좌표계로 변환
        rotated_x = (
            rotation_matrix[0, 0] * dx
            + rotation_matrix[1, 0] * dy
            + rotation_matrix[2, 0] * dz
        )
        rotated_y = (
            rotation_matrix[0, 1] * dx
            + rotation_matrix[1, 1] * dy
            + rotation_matrix[2, 1] * dz
        )
        rotated_z = (
            rotation_matrix[0, 2] * dx
            + rotation_matrix[1, 2] * dy
            + rotation_matrix[2, 2] * dz
        )

        # 기본 타원체의 거리장
        ellipsoid_field = (
            (rotated_x / radii[0]) ** 2
            + (rotated_y / radii[1]) ** 2
            + (rotated_z / radii[2]) ** 2
        )

        # 부드러운 3차원 랜덤 노이즈 생성
        noise = rng.normal(size=tuple(grid_shape))
        smooth_noise = gaussian_filter(
            noise,
            sigma=noise_sigma,
            mode="reflect",
        )

        noise_mean = np.mean(smooth_noise)
        noise_std = np.std(smooth_noise)

        if np.isclose(noise_std, 0.0):
            continue

        smooth_noise = (smooth_noise - noise_mean) / noise_std

        # 타원체 표면을 노이즈로 변형
        random_field = ellipsoid_field + noise_strength * smooth_noise
        mass_mask_3d = random_field <= 1.0

        # 정육면체 경계와 일정한 간격 유지
        if margin_voxels > 0:
            mass_mask_3d[:margin_voxels, :, :] = False
            mass_mask_3d[-margin_voxels:, :, :] = False

            mass_mask_3d[:, :margin_voxels, :] = False
            mass_mask_3d[:, -margin_voxels:, :] = False

            mass_mask_3d[:, :, :margin_voxels] = False
            mass_mask_3d[:, :, -margin_voxels:] = False

        # 내부에 생긴 작은 구멍 제거
        mass_mask_3d = binary_fill_holes(mass_mask_3d)

        # 연결 요소 구분
        labeled_mask, number_of_components = label(
            mass_mask_3d,
            structure=six_neighbor_structure,
        )

        if number_of_components == 0:
            continue

        # 가장 큰 연결 요소만 유지
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0

        largest_component = np.argmax(component_sizes)
        mass_mask_3d = labeled_mask == largest_component

        # 연결 요소 처리 이후 생길 수 있는 내부 구멍을 다시 제거
        mass_mask_3d = binary_fill_holes(mass_mask_3d)

        volume_ratio = np.count_nonzero(mass_mask_3d) / mass_mask_3d.size

        if min_volume_ratio <= volume_ratio <= max_volume_ratio:
            return mass_mask_3d.ravel(order="C")

    raise RuntimeError(
        f"{max_attempts}회 안에 부피 비율 "
        f"{min_volume_ratio:.1%}~{max_volume_ratio:.1%}를 만족하는 "
        "무작위 덩어리를 생성하지 못했습니다."
    )


# 최초 가지 설정
def select_initial_trunk(
    organ_mask,
    voxel_coords,
    initial_length=1.2,
):
    """
    랜덤 덩어리의 위쪽 표면에서 시작하여
    덩어리 중심 방향으로 향하는 초기 노드 2개를 선택한다.

    organ_mask:
        True가 덩어리 내부인 1차원 Boolean 배열
    voxel_coords:
        main.py에서 생성한 (N, 3) voxel 중심 좌표
    initial_length:
        초기 가지의 목표 길이(cm)
    return:
        start_pos, end_pos
    """

    organ_voxels = voxel_coords[organ_mask]

    if len(organ_voxels) < 2:
        raise ValueError("초기 가지를 생성할 만큼 장기 voxel이 충분하지 않습니다.")

    # 덩어리 전체의 중심
    organ_centroid = np.mean(organ_voxels, axis=0)

    # z 좌표가 가장 큰 위쪽 표면
    maximum_z = np.max(organ_voxels[:, 2])
    top_voxels = organ_voxels[
        np.isclose(
            organ_voxels[:, 2],
            maximum_z,
        )
    ]

    # 위쪽 표면 voxel 중 덩어리의 x-y 중심과 가장 가까운 점 선택
    horizontal_distances = np.linalg.norm(
        top_voxels[:, :2] - organ_centroid[:2],
        axis=1,
    )

    start_pos = top_voxels[np.argmin(horizontal_distances)]

    # 시작점에서 덩어리 중심을 향하는 방향
    inward_vector = organ_centroid - start_pos
    inward_length = np.linalg.norm(inward_vector)

    if np.isclose(inward_length, 0.0):
        raise RuntimeError("초기 가지 방향을 계산할 수 없습니다.")

    inward_unit_vector = inward_vector / inward_length

    # 목표 초기 노드 위치
    target_end_pos = start_pos + inward_unit_vector * initial_length

    # 목표 위치에서 가장 가까운 장기 내부 voxel 선택
    organ_tree = KDTree(organ_voxels)
    _, nearest_index = organ_tree.query(target_end_pos)

    end_pos = organ_voxels[nearest_index]

    if np.allclose(start_pos, end_pos):
        raise RuntimeError(
            "초기 가지의 두 노드가 같은 voxel에 배치되었습니다. "
            "initial_length를 크게 설정하세요."
        )

    return start_pos, end_pos
