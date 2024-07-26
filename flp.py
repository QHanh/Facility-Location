from scipy.optimize import linprog
import numpy as np

location = ["D3", "D3.5", "D5", "D6", "D7", "D8", "D9", "TV", "C1", "C3", "C4", "C5"]

D = np.array([
    [0, 5, 10, 80, 15, 80, 60, 30, 95, 90, 85, 80],
    [5, 0, 10, 85, 20, 85, 65, 35, 100, 95, 90, 85],
    [10, 5, 0, 85, 5, 85, 20, 35, 115, 100, 95, 90],
    [80, 85, 85, 0, 85, 10, 45, 40, 90, 85, 80, 75],
    [15, 5, 5, 85, 0, 80, 5, 35, 110, 105, 100, 95],
    [80, 85, 85, 10, 80, 0, 40, 50, 80, 75, 70, 65],
    [60, 65, 20, 45, 5, 40, 0, 25, 90, 85, 80, 75],
    [30, 35, 35, 40, 35, 50, 25, 0, 80, 75, 70, 65],
    [95, 100, 115, 90, 110, 80, 90, 80, 0, 5, 10, 15],
    [90, 95, 100, 85, 105, 75, 85, 75, 5, 0, 5, 10],
    [85, 90, 95, 80, 100, 70, 80, 70, 10, 5, 0, 5],
    [80, 85, 90, 75, 95, 65, 75, 65, 15, 10, 5, 0]
])

h = np.array([10, 7, 18, 25, 25, 40, 30, 8, 10, 10, 10, 10])
p = 3

num_locations = len(location)

# Hệ số hàm mục tiêu
c = (h.reshape(-1, 1) * D).flatten()

# Ràng buộc
A_eq = np.zeros((2 * num_locations, num_locations ** 2))
b_eq = np.zeros(2 * num_locations)

# Ràng buộc: sum(Y[i][i]) == p
for i in range(num_locations):
    A_eq[0, i * num_locations + i] = 1
b_eq[0] = p

# Ràng buộc: sum(Y[i][j]) == 1 for each i
for i in range(num_locations):
    A_eq[i + 1, i * num_locations:(i + 1) * num_locations] = 1
    b_eq[i + 1] = 1

# Ràng buộc: Y[i][j] <= Y[j][j] for all i, j
A_ub = np.zeros((num_locations ** 2, num_locations ** 2))
b_ub = np.zeros(num_locations ** 2)

for i in range(num_locations):
    for j in range(num_locations):
        A_ub[i * num_locations + j, i * num_locations + j] = 1
        A_ub[i * num_locations + j, j * num_locations + j] = -1

# Biến
bounds = [(0, 1) for _ in range(num_locations ** 2)]

# Giải bài toán
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# In kết quả
print("Status:", res.message)
print("Objective: ", res.fun)

Y = res.x.reshape(num_locations, num_locations)
for i in range(num_locations):
    for j in range(num_locations):
        print(f"Y[{location[i]}][{location[j]}] = {Y[i, j]}")
