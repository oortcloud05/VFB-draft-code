import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from input import G
from freespace import update_freespace
from angle import bifurcation
from limit_angle import calculate_branch_angle


# 1. 3D 공간 및 반복 설정
space_size = (10, 10, 10)
voxel_resolution = 0.1
max_iterations = 30

# 2. voxel 중심 좌표 계산
x = np.arange(0, space_size[0], voxel_resolution) + voxel_resolution / 2
y = np.arange(0, space_size[1], voxel_resolution) + voxel_resolution / 2
z = np.arange(0, space_size[2], voxel_resolution) + voxel_resolution / 2

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
voxel_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


# 3. free space mask 생성
freespace_mask = np.ones(voxel_coords.shape[0], dtype=bool)


# 4. 초기 가지 구조를 free space에 업데이트 -> 각 가지를 free space에서 반복하여 제거
for start_node, end_node, attr in G.edges(data=True):
    start_pos = np.array(G.nodes[start_node]["pos"])  # 시작 노드 좌표
    end_pos = np.array(G.nodes[end_node]["pos"])  # 끝 노드 좌표
    diameter = attr.get("diameter", 1.0)  # 기본 diameter 설정 (없으면 1.0)

    freespace_mask = update_freespace(
        start_pos, end_pos, diameter, voxel_coords, voxel_resolution, freespace_mask
    )


## 초기 구조와 VFB에서 새로 생성된 branch를 구분하기 위한 노드 목록
initial_node_ids = set(G.nodes)


# 5. 반복하며 VFB method 적용 및 free space update
for _ in range(max_iterations):
    # 새로운 가지의 ending point 위치 계산
    (
        selected_point_pos,
        previous_node_pos,
        new_node_1_pos,
        new_node_2_pos,
        new_diameter,
    ) = bifurcation(G, freespace_mask, voxel_coords, voxel_resolution, space_size)

    if selected_point_pos is None:
        print("No more branching possible.")
        break

    selected_node_id = None
    previous_node_id = None

    for node, data in G.nodes(data=True):
        if np.allclose(data["pos"], selected_point_pos, atol=1e-5):
            selected_node_id = node
        if np.allclose(data["pos"], previous_node_pos, atol=1e-5):
            previous_node_id = node

    if selected_node_id is None or previous_node_id is None:
        print("Error: Unable to find node IDs in G.")
        break

    new_node_1 = max(G.nodes) + 1
    new_node_2 = new_node_1 + 1

    # 새로운 가지를 그래프에 추가
    G.add_node(new_node_1, pos=tuple(new_node_1_pos))
    G.add_node(new_node_2, pos=tuple(new_node_2_pos))
    G.add_edge(selected_node_id, new_node_1, diameter=new_diameter)
    G.add_edge(selected_node_id, new_node_2, diameter=new_diameter)

    # free space update
    freespace_mask = update_freespace(
        tuple(G.nodes[selected_node_id]["pos"]),
        tuple(G.nodes[new_node_1]["pos"]),
        new_diameter,
        voxel_coords,
        voxel_resolution,
        freespace_mask,
    )
    freespace_mask = update_freespace(
        tuple(G.nodes[selected_node_id]["pos"]),
        tuple(G.nodes[new_node_2]["pos"]),
        new_diameter,
        voxel_coords,
        voxel_resolution,
        freespace_mask,
    )


# 6. 시각화
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")

# 자유 공간과 가지 표시
occupied_voxels = voxel_coords[~freespace_mask]
ax.scatter(
    occupied_voxels[:, 0],
    occupied_voxels[:, 1],
    occupied_voxels[:, 2],
    c="gray",
    s=5,
    alpha=0.5,
)

# G의 노드 & 엣지 좌표 가져오기
for start_node, end_node in G.edges():
    start_pos = np.array(G.nodes[start_node]["pos"])
    end_pos = np.array(G.nodes[end_node]["pos"])

    ax.plot(
        [start_pos[0], end_pos[0]],
        [start_pos[1], end_pos[1]],
        [start_pos[2], end_pos[2]],
        "r-",
        linewidth=2,
    )

# 노드 그리기 (붉은색 점)
node_positions = np.array([G.nodes[n]["pos"] for n in G.nodes()])
ax.scatter(
    node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], c="red", s=20
)

# 전체 공간이 표시되도록 x, y, z 축 고정
ax.set_xlim([0, space_size[0]])
ax.set_ylim([0, space_size[1]])
ax.set_zlim([0, space_size[2]])


# 축의 비율을 동일하게 설정 (equal aspect ratio)
def set_axes_equal(ax):
    """x, y, z 축을 동일한 크기로 설정"""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    span = limits[:, 1] - limits[:, 0]
    mean = np.mean(limits, axis=1)
    max_span = max(span)

    ax.set_xlim(mean[0] - max_span / 2, mean[0] + max_span / 2)
    ax.set_ylim(mean[1] - max_span / 2, mean[1] + max_span / 2)
    ax.set_zlim(mean[2] - max_span / 2, mean[2] + max_span / 2)


set_axes_equal(ax)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend().remove()


# 번외) branch 각도 분포 히스토그램


## VFB에서 새로 생성된 모든 자녀 branch의 최종 각도 수집
branch_angles = []

for selected_node_id, new_node_id in G.edges():
    # 초기 입력 구조에 포함된 edge는 제외
    if new_node_id in initial_node_ids:
        continue

    # selected node의 부모 노드 찾기
    previous_node_id = next(
        G.predecessors(selected_node_id),
        None,
    )

    # 부모 branch가 없으면 각도를 계산할 수 없음
    if previous_node_id is None:
        continue

    previous_node_pos = np.array(G.nodes[previous_node_id]["pos"])
    selected_point_pos = np.array(G.nodes[selected_node_id]["pos"])
    new_node_pos = np.array(G.nodes[new_node_id]["pos"])

    angle_deg = calculate_branch_angle(
        previous_node_pos,
        selected_point_pos,
        new_node_pos,
    )

    # 길이가 0인 branch에서 발생하는 nan 제외
    if np.isfinite(angle_deg):
        branch_angles.append(angle_deg)

branch_angles = np.asarray(branch_angles)

## 히스토그램 작성
if len(branch_angles) > 0:
    angle_bins = np.arange(0, 185, 2)

    # 전체 branch 중 각 구간이 차지하는 비율
    histogram_weights = np.ones(len(branch_angles)) / len(branch_angles) * 100

    hist_fig, hist_ax = plt.subplots(figsize=(9, 6))

    hist_ax.hist(
        branch_angles,
        bins=angle_bins,
        weights=histogram_weights,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )

    # 제한 각도 표시
    hist_ax.axvline(
        60,
        color="red",
        linestyle="--",
        linewidth=2,
        label="60-degree limit",
    )

    hist_ax.set_xlim(0, 180)
    hist_ax.set_ylim(0, 100)
    hist_ax.set_xticks(np.arange(0, 181, 20))

    hist_ax.set_xlabel("Branch angle (degrees)")
    hist_ax.set_ylabel("Percentage of branches (%)")
    hist_ax.set_title("Final Branch Angle Distribution")

    hist_ax.grid(
        axis="y",
        alpha=0.3,
    )
    hist_ax.legend()

    hist_fig.tight_layout()

    print(f"Number of branches: {len(branch_angles)}")
    print(f"Mean angle: {np.mean(branch_angles):.2f} degrees")
    print(f"Maximum angle: {np.max(branch_angles):.2f} degrees")
    print(f"Branches over 60 degrees: {np.sum(branch_angles > 60.0)}")

else:
    print("No valid branch angles were collected.")


plt.show()
