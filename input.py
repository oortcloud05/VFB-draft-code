import numpy as np
import networkx as nx


# 모든 단위는 cm scale

# 노드의 좌표 (예: 5개의 노드, 0부터 4까지)
node_coords = np.array([
    [5.05, 5.05, 0.05],    
    [5.05, 5.05, 1.25],    
    [8.05, 5.05, 3.05],    
    [2.05, 5.05, 3.05],    
    [4.95, 5.05, 3.85],
    [7.05, 6.05, 3.95]    
])

# 방향성 있는 인접 행렬 (5x5, 노드 0부터 4까지)
adj_matrix = np.array([
    [0, 1, 0, 0, 0, 0],  
    [0, 0, 1, 1, 0, 0],  
    [0, 0, 0, 0, 1, 1],  
    [0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

# 각 edge의 diameter 정보 (edge가 없으면 0)
diameter_matrix = np.array([
    [0, 0.75, 0, 0, 0, 0],
    [0, 0, 0.38, 0.48, 0, 0],
    [0, 0, 0, 0, 0.36, 0.35],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

# ✅ 방향 그래프 생성
G = nx.DiGraph()
num_nodes = adj_matrix.shape[0]
G.add_nodes_from(range(num_nodes))

# ✅ 노드에 위치 정보 추가 (tuple로 변환하여 저장)
for i, coord in enumerate(node_coords):
    G.nodes[i]['pos'] = tuple(coord)  # numpy 배열이 아니라 tuple 형태로 저장

# ✅ 간선 추가 (diameter 포함)
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj_matrix[i, j] == 1:  # 연결된 경우
            diameter = diameter_matrix[i, j] if diameter_matrix[i, j] > 0 else 1.0  # 기본값 1.0
            G.add_edge(i, j, diameter=diameter)

# ✅ `main.py`에서 가져다 사용할 수 있도록 설정
__all__ = ['G']
