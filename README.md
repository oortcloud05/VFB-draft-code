### V4 기준 수정 가능한 parameter

1. 실행 관련 (main.py)  
space_size (line 16) : 전체 계산 공간 크기  
** space_size에서 축 길이 바꾸면 장기 모양도 극단적으로 생성 가능  
voxel_resolution (line 20) : voxel 한 변의 길이 (작을수록 정확하게 연산)  
max_iterations (line 21) : 최대 분기 실행 횟수  

3. 장기 모양 (generate_random_mass, line 49)  
min_volume_ratio, max_volume_ratio : 생성된 장기 영역이 전체 공간에서 차지해야 하는 부피 비율 범위  
noise_strength : 장기 표면 불규칙성 강도  
noise_smoothness : 표면 noise가 얼마나 부드럽게 이어지는지  
margin_voxels : 장기가 전체 계산 공간의 외벽에 닿지 않도록 비워 두는 voxel 층의 수  
seed : 정수로 설정하면 같은 랜덤 결과 재현 (현재는 None, 매번 결과 다르게)  

4. 장기 결정 (random_mass.py)  
max_attempts (line 15) : 장기 생성 시도 횟수  
center_shift (line 80) : 장기 중심 위치 변화 (low, high 값 조정)  
radius_fractions (line 92) : 타원체 반지름 범위  

5. 초기 가지  
initial_length (main.py, line 70) : 최초 가지의 목표 길이  
diameter (main.py, line 89) : 최초 가지의 지름  
move_distance (angle.py, line 152) : 최초 가지로부터 처음으로 분기되는 두 가지의 목표 길이  

6. 분기 시 가지 배수 (angle.py)  
(line 98) : 가지 직경 배수  
(line 155) : 가지 길이 배수  

7. terminal condition (terminal_condition.py)  
min_branch_length (line 8) : 새 가지의 최소 길이  
min_branch_diameter (line 18) : 새 가지의 최소 직경  

8. 최대 분기각 (angle.py)  
max_angle_deg (line 177, 183) : 다음 가지의 최대 분기각  

9. 시각화 전용 (main.py)  
그림 크기, 히스토그램 구간 폭, 표시할 장기 voxel 최대 개수 등  





