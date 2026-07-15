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
