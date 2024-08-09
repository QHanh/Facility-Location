from pulp import *

location = ["D3", "D3.5", "D5", "D6", "D7", "D8", "D9", "TV", "C1", "C3", "C4", "C5"]
D = dict(zip(location, [dict(zip(location, [0, 5, 10, 80, 15, 80, 60, 30, 95, 90, 85, 80])),
                        dict(zip(location, [5, 0, 10, 85, 20, 85, 65, 35, 100, 95, 90, 85])),
                        dict(zip(location, [10, 5, 0, 85, 5, 85, 20, 35, 115, 100, 95, 90])),
                        dict(zip(location, [80, 85, 85, 0, 85, 10, 45, 40, 90, 85, 80, 75])),
                        dict(zip(location, [15, 5, 5, 85, 0, 80, 5, 35, 110, 105, 100, 95])),
                        dict(zip(location, [80, 85, 85, 10, 80, 0, 40, 50, 80, 75, 70, 65])),
                        dict(zip(location, [60, 65, 20, 45, 5, 40, 0, 25, 90, 85, 80, 75])),
                        dict(zip(location, [30, 35, 35, 40, 35, 50, 25, 0, 80, 75, 70, 65])),
                        dict(zip(location, [95, 100, 115, 90, 110, 80, 90, 80, 0, 5, 10, 15])),
                        dict(zip(location, [90, 95, 100, 85, 105, 75, 85, 75, 5, 0, 5, 10])),
                        dict(zip(location, [85, 90, 95, 80, 100, 70, 80, 70, 10, 5, 0, 5])),
                        dict(zip(location, [80, 85, 90, 75, 95, 65, 75, 65, 15, 10, 5, 0]))]))

h = {"D3": 10, "D3.5": 7, "D5": 18, "D6": 25, "D7": 25, "D8": 40, "D9": 30, "TV": 8, "C1": 10, "C3": 10, "C4": 10, "C5": 10}
p = 3

Y = LpVariable.dicts("Y", (location, location), cat="Binary", lowBound=0, upBound=1)
prob = LpProblem("P Median", LpMinimize)

# Hàm mục tiêu
prob += sum(sum(h[i] * D[i][j] * Y[i][j] for j in location) for i in location)

# Ràng buộc
prob += sum(Y[i][i] for i in location) == p

for i in location:
    prob += sum(Y[i][j] for j in location) == 1

for i in location:
    for j in location:
        prob += Y[i][j] <= Y[j][j]

prob.writeLP("P-median.lp")

prob.solve()

print("Status:", LpStatus[prob.status])


for v in prob.variables():
    print(v.name.replace(" ", "_"), "=", v.varValue)
    
print("Objective: ", value(prob.objective))
