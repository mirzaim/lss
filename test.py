import linear_system_solver as lss

a = [[2, -4, 5],
     [4, -1, 0],
     [-2, 2, -3]]

b = [[-33],
     [-5],
     [19]]

ls = lss.LinearSystemSolver(a, b)
print(ls.get_steps_to_solve())
print(ls.get_reduced_echelon_form())

print("#" * 79)

a = [[0, 3, -6, 6, 4],
     [3, -7, 8, -5, 8],
     [3, -9, 12, -9, 6]]

b = [[-5],
     [9],
     [15]]

ls = lss.LinearSystemSolver(a, b)
print(ls.get_reduced_echelon_form())
print(ls.result_matrix())
print(ls.get_result_string())
