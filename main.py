import linear_system_solver as lss

# write coefficient matrix A here.
A = [[1, 3, 2, -4, 3],
     [-2, -1, 2, 6, 4],
     [0, -1, 3, -5, 1],
     [3, -4, 2, 5, -7],
     [1, 2, -8, 6, 1]]

# write constant b vector here.
b = [[-3],
     [19],
     [-2],
     [-11],
     [4]]

ls = lss.LinearSystemSolver(A, b)
print(ls.get_steps_to_solve())
print(ls.get_reduced_echelon_form())
print(ls.get_result_string())
