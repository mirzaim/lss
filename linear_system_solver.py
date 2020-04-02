import numpy as np


class LinearSystemSolver:

    def __init__(self, a, b=None):
        """"""
        if b is None:
            b = np.zeros((a.shape[0], 1))

        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.augmented_matrix = np.column_stack((self.a, self.b))
        self.pivot_columns = []
        self.steps_to_solve = []

        self.echelon_form()
        self.augmented_matrix = np.around(self.augmented_matrix, 3)

    def rank_a(self):
        """return rank coefficient matrix A"""
        return self.a.shape[1] - self.dim_nul_a()

    def dim_nul_a(self):
        """return dimension of null space A"""
        temp = LinearSystemSolver(self.a)
        return len(temp.result_matrix()) - 1

    def row_replacement(self, k, i, j):
        """sum row i with k times of j in augmented matrix"""
        self.augmented_matrix[i] += self.augmented_matrix[j] * k

    def row_scaling(self, k, i):
        """scale row i, k times in augmented matrix"""
        self.augmented_matrix[i] *= k

    def row_interchange(self, i, j):
        """change place of row i and j in augmented matrix"""
        self.augmented_matrix[[i, j]] = self.augmented_matrix[[j, i]]

    def echelon_form(self):
        """do row operation and change matrix to echelon form"""
        current_pivot_column = 0
        self.steps_to_solve.append(self.augmented_matrix.__str__())
        for i in range(self.augmented_matrix.shape[0]):
            while self.augmented_matrix[i][current_pivot_column] == 0:
                for bottom in range(i + 1, self.augmented_matrix.shape[0]):
                    if self.augmented_matrix[bottom][current_pivot_column] != 0:
                        self.row_interchange(i, bottom)
                        self.steps_to_solve.append(self.augmented_matrix.__str__())
                        break
                else:
                    current_pivot_column += 1

            for other in range(self.augmented_matrix.shape[0]):
                if i == other:
                    continue
                times = (self.augmented_matrix[other][current_pivot_column] /
                         self.augmented_matrix[i][current_pivot_column])
                self.row_replacement(-1 * times, other, i)

            self.row_scaling((1 / self.augmented_matrix[i][current_pivot_column]), i)
            self.pivot_columns.append(current_pivot_column)
            self.steps_to_solve.append(self.augmented_matrix.__str__())
            current_pivot_column += 1

    def get_reduced_echelon_form(self):
        """return augmented matrix"""
        return self.augmented_matrix.copy()

    def get_steps_to_solve(self):
        """get steps in string that passed to reach result."""
        steps = "#### Steps:\n"
        for i in range(len(self.steps_to_solve)):
            steps += f"{i + 1})\n"
            steps += (self.steps_to_solve[i].__str__() + "\n")
        steps += "- " * 39
        return steps

    def is_consistent(self):
        """return solution to "is consistent?" question"""
        # Check last column in augmented matrix is pivot column or not
        return (self.augmented_matrix.shape[1] - 1) not in self.pivot_columns

    def result_matrix(self):
        """return vectors that form result."""
        vectors = []
        for j in list(set(range(self.augmented_matrix.shape[1])) - set(self.pivot_columns)):
            temp = []
            for i in range(self.augmented_matrix.shape[1] - 1):
                if i in self.pivot_columns:
                    temp.append(self.augmented_matrix[self.pivot_columns.index(i)][j])
                elif i == j:
                    temp.append(-1)
                else:
                    temp.append(0)
            vectors.append(temp[:])
        return vectors

    def get_result_string(self):
        """return string that show the result of system."""
        if not self.is_consistent():
            return "INCONSISTENT"

        result_string = "result: x = "
        result_vector = self.result_matrix()

        result_string += tuple(result_vector.pop()).__str__() + " "
        for i in range(len(result_vector)):
            result_string += "+ " + tuple(result_vector[i]).__str__() + f".t{i + 1} "

        return result_string
