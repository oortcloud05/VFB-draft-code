### V1(original)

최종 말단점 결정 시 voxel 경계가 아닌 voxel 중심점을 기준으로 반올림하도록 수식 수정  
(angle.py line 99-105)



### V2(avoid_intersection)

1. avoid intersection 구현 시도

	- intersection.py  
 	centroid 계산 후, 후보 voxel 여러 개 탐색  
	angle(최초 생성 가지와 각도 차이), endpoint(최초 말단점과 거리 차이), length(최초 생성 가지와 길이 차이)  
	penalty 점수화하여 최적 말단점 선정
	
	- angle.py에 avoid_intersection() 추가

2. freespace.py 중간 주석

	기존 코드는 생성된 가지 위에 점을 voxel_resolution 간격으로 찍은 뒤,  
	각 점에서 가지 반지름을 반지름으로 하는 구를 그리고,  
	구체 내부에 중심점이 있는 voxel을 false로 마스킹  
	-> 가지 주변 voxel과 가지의 수직 거리를 반지름과 직접 비교하여 마스킹 시도  
	but 계산량이 너무 많아질 것 같음

3. freespace_new.py

	2번의 계산량 많은 문제 해결: edge 주변의 voxel에 대해서만 계산 (bounding box)  
	좀 더 정확한 마스킹이긴 한데 어차피 resolution이 충분히 작으면 큰 영향 X
	
	** 기존 방법도 점을 찍고 나서 모든 voxel을 계산함. bounding box 활용하면 계산량 훨씬 줄어들 듯



### V3(main)

1. 분기 종료 지점 수정 (angle.py)

	기존 코드는 가장 작은 index 가진 node만 검사, 분기 불가하면 전체 종료  
	-> line 68부터 반복문 전체  
	node를 index 순서대로 검사하면서 최초로 가능한 분기 시행  

2. limit angle 반영 (limit_angle.py)

	- limit_angle()  
	부모 가지와 새 가지 사이 각도가 60°보다 크면 60° 방향으로 보정, 새 가지 길이는 유지
	
	- calculate_branch_angle()  
	이후 각도 분포 히스토그램 작성을 위해 최종 그래프의 실제 분기각 측정

3. terminal condition 반영 (terminal_condition.py)

	- (#1) 가지 길이 제한  
	branch_length_valid(), 최소 가지 길이 가능
	
	- (#2) 가지 직경 제한  
	branch_diameter_valid(), 최소 가지 지름 설정 가능
	
	- (#3) ending point가 최외곽 voxel에 도달  
	ending_point_at_boundary()

4. angle.py에 추가

	- 최대각(60도) 보정: line 148-160
	
	- (#1) 가지 길이 제한: line 170-180
	
	- (#2) 가지 직경 제한: line 86-93
	
	- (#3) ending point가 최외곽 voxel에 도달: line 40-45  
	처음부터 최외곽 voxel은 후보점에서 제외

5. main.py 수정

	- 최대 분기 횟수 설정 가능하도록 수정  
	line 17-18: max_iteration 설정, completed_iteration(실제 완료된 반복 횟수) 측정
	
	- 연산 시간 측정  
	line 12: 측정 시작  
	line 107-122: 측정 종료, 시간 출력
	
	- 각도 분포 히스토그램 작성  
	line 185-272



### V3-2(node_center)

limit angle 반영 시 voxel 후보를 먼저 찾아서 예외 없이 적용되도록 함  
but 우선순위 문제로 보류



### V4(random_mass)

1. 무작위 장기 형상 생성 (random_mass.py)

	- generate_random_mass()  
	무작위 크기·중심·회전을 가진 타원체 + Gaussian noise로 표면 변형
	
	- select_initial_trunk()  
	장기 상단 표면에서 시작 voxel 선택, 장기 중심 방향으로 초기 혈관 끝점 선택
	(기존 코드에서 input 역할)

2. main.py 수정

	line 48-98: 랜덤으로 장기 형상 생성  
	line 202-231: 생성된 장기 형태 시각화

3. terminal condition #3 관련 수정 (terminal_condition.py)

	ending_point_at_boundary(): 기존 코드에서는 정육면체 공간 기준 최외곽 voxel 검사  
	-> 새로 생성된 불규칙한 장기 모양 기준 최외곽 voxel 검사  
	말단의 상하좌우앞뒤 6개 인접 voxel이 장기 밖인지 검사하도록 수정

4. initial trunk 관련 수정

	input.py에서 임의로 최초 node를 설정할 수 없어짐  
	-> select_initial_trunk()로 해결 but 가지 길이 배수 때문에 최초 가지가 짧으면 분기가 너무 빨리 끝남  
	-> is_initial=True 인 최초 분기만 길이를 고정값으로 입력


