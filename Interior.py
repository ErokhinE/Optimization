import re
import sys
import numpy as np

"""
Example of input:
5 4 0 0 0 0
6 4 1 0 0 0|1 2 0 1 0 0|-1 1 0 0 1 0|0 1 0 0 0 1
22 6 1 2
1 1 12 3 1 1
3

"""


def greeting():
    print("""The input contains:
• A vector of coefficients of the objective function - C.
• A matrix of coefficients of the constraint function - A.
• A vector of right-hand side numbers - b.
• A vector of initial trial solution
• The approximation accuracy ϵ
============================
vector C input format:
x1 x2 x3... xn
for example:
5 4 0 0 0 0
============================
matrix A input format:
x1 x2 x3 x4 x5 x6|x1 x2 x3 x4 x5 x6|x1 x2 x3 x4 x5 x6|x1 x2 x3 x4 x5 x5
for example:
6 4 1 0 0 0|1 2 0 1 0 0|-1 1 0 0 1 0|0 1 0 0 0 1
============================
A vector of right-hand side numbers b input format:
sol1 sol2 sol3 sol3
for example:
22 6 1 2
============================
A vector of initial trial solution input format:
x1_0 x2_0 x3_0 x4_0 x5_0 x6_0
for example:
1 1 12 3 1 1
============================
""")


def simplex_method(C, A, b, accur):
    def check_matrix(S):
        return any(m < 0 for m in S[0])

    def check_matrix_size(C, A, b):
        if len(C) != len(A[0]) or len(A) != len(b):
            print("The method is not applicable! Matrix and vector dimensions do not match.")
            sys.exit(1)

    def check_infeasibility(b):
        if any(val < 0 for val in b):
            print("The method is not applicable! The problem is infeasible.")
            sys.exit(1)

    def create_titles(count_of_x_variables, count_of_equations):
        var_rows = ["z"] + [f"s{number + 1}" for number in range(count_of_equations)]
        var_colums = [f"x{number + 1}" for number in range(count_of_x_variables)]
        var_colums.extend(var_rows[1:])
        var_colums.append("b")
        return var_rows, var_colums

    def Iteration(S, var_colums, var_rows):
        i = 0
        j = -1
        min_value = S[0][0]
        min_ratio = sys.maxsize
        # find min negative value in z-row
        for m in range(1, len(S[0])):
            if S[0][m] < min_value:
                min_value = S[0][m]
                i = m
        # find element with min ratio
        for n in range(1, len(S)):
            if S[n][i] == 0:
                continue
            ratio = S[n][-1] / S[n][i]
            if 0 < ratio < min_ratio:
                min_ratio = ratio
                j = n
        # make 0 another values in column
        temp = S[j][i]

        for m in range(len(S[j])):
            S[j][m] = S[j][m] / temp

        for n in range(len(S)):
            if n == j:
                continue
            k = S[n][i]
            for m in range(len(S[n])):
                S[n][m] -= k * S[j][m]
        # change row variable
        var_rows[j] = var_colums[i]
        return S, var_rows

    # check method applicable or not
    check_matrix_size(C, A, b)
    check_infeasibility(b)

    # create titles for table
    var_rows, var_colums = create_titles(len(C), len(A))

    # create table S
    S = [[]]

    # fill z-row
    for m in range(len(C)):
        S[0].append(-1 * C[m])

    S[0].append(0)

    # fill other rows
    for n in range(len(A)):
        def_row = []
        for m in range(len(C)):
            def_row.append(A[n][m])
        def_row.append(b[n])
        S.append(def_row)

    # loop of iterations
    while check_matrix(S):
        S, var_rows = Iteration(S, var_colums, var_rows)

    # prepare the result
    result = {}
    for variable_index in range(len(var_rows)):
        variable = var_rows[variable_index]
        if "x" in variable:
            result[variable] = format(S[variable_index][-1], f".{accur}f")

    result["Solution"] = format(S[0][-1], f".{accur}f")

    for i in range(1, len(C) + 1):
        if f"x{i}" not in result:
            result[f"x{i}"] = format(0, f".{accur}f")

    return result


def check_partial_solution(X, A, b):
    for i in range(len(A)):
        row_sum = np.sum(np.multiply(A[i], X))
        if not np.isclose(row_sum, b[i], rtol=1e-7):
            return False
    return True


def input_vector(prompt, is_integer=False):
    while True:
        input_str = input(prompt)
        if is_integer:
            if re.match(r'^[+-]?\d+$', input_str):
                return int(input_str)
        else:
            if all(re.match(r'^[+-]?\d+(\.\d+)?$', item) for item in input_str.split()):
                return [float(item) for item in input_str.split()]

        print("Invalid input. Please enter a valid value.")


def input_matrix(prompt):
    matrix = []
    while True:
        input_str = input(prompt)

        if not all(re.match(r'^[+-]?\d+(\.\d+)?$', item) for row in input_str.split('|') for item in row.split()):
            print("Invalid input. Please enter a valid matrix.")
            continue

        rows = [list(map(float, row.split())) for row in input_str.split('|')]
        if all(len(row) == len(rows[0]) for row in rows):
            return rows
        print("Invalid input. Please enter a valid matrix.")


def inp():
    # input data and check correct data or not
    print("Write C vector in requirements format\n")
    C = input_vector("")
    count_of_all_variables = len(C)
    print("Write A matrix in requirements format\n")
    A = input_matrix("")

    print("Write b vector in requirements format\n")
    b = input_vector("")
    print("Write feasible solution vector in requirements format\n")
    feasible_solution = input_vector("")
    if not check_partial_solution(feasible_solution, A, b):
        print("Partial solution does not satisfy constraints.")
        return

    accur = input_vector("Write the accuracy of the calculations: ", is_integer=True)
    print("============================")
    print("C vector:", C)
    print("Matrix A:", A)
    print("b vector:", b)
    print("matrix with feasible solutions: ", feasible_solution)
    count_of_equations = len(A)
    return C, A, b, feasible_solution, accur, count_of_all_variables, count_of_equations


def multiply_and_sum_vectors(vector1, vector2):
    # Check if the vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    # Multiply corresponding elements of the vectors and sum the results
    result = sum(x * y for x, y in zip(vector1, vector2))

    return result


def InterPoint(C, A, accur, alpha, feasible_solution, count_of_all_variables):
    # make A and C, feasible_solution numpy arrays
    C_np = np.transpose(np.array(C))
    A_np = np.array(A)
    X = np.array(feasible_solution)
    print("numpy matrix A:", A_np)
    print("numpy matrix C:", C_np)
    print("numpy matrix feasible solution:", X)
    # make Identity matrix I
    I = np.eye(count_of_all_variables)
    print("Identity matrix I: ", I)
    iteration = 1
    while True:
        print(f"\n\n==== iteration{iteration} ====")
        # make matrix D
        D = np.diag(X)
        print("Diag matrix D: ", D)
        # calculate new A (const A * dinamic D)
        new_A = np.dot(A_np, D)
        print("new matrix A: ", new_A)
        # calculate new C (dinamic D * const C)
        new_C = np.dot(D, C_np)
        print("new matrix C: ", new_C)
        # calculate A transpose
        new_A_T = np.transpose(new_A)
        # calculate P = I - A_T*(A*A_T)^-1 * A use temp variables
        AA_T = np.dot(new_A, new_A_T)
        print("matrix AA_T ", AA_T)
        AA_T_inv = np.linalg.inv(AA_T)
        A_T__AA_T_inv = np.dot(new_A_T, AA_T_inv)
        A_T__AA_T_inv_A = np.dot(A_T__AA_T_inv, new_A)
        P = I - A_T__AA_T_inv_A
        print("matrix P: ", P)
        # calculate Cp
        Cp = np.dot(P, new_C)
        print("matrix Cp: ", Cp)
        # find v - absolute value, which min in Cp
        v = np.absolute(np.min(Cp))
        # find X_dot
        X_dot = np.add(np.ones(count_of_all_variables, float), (alpha / v) * Cp)
        print("X_dot matrix: ", X_dot)
        # calculate new_X
        new_X = np.dot(D, X_dot)
        print("new X: ", new_X)

        if np.linalg.norm(np.subtract(X, new_X), ord=2) <= 1 / 10 ** accur:
            print("STOP")
            print("Answer: ", new_X)
            return new_X
        iteration += 1
        X = new_X


greeting()  # function with rules

C, A, b, feasible_solution, accur, count_of_all_variables, count_of_equations = inp()  # input data from console

simplex_ans = simplex_method(C, A, b, accur)
with_05 = InterPoint(C, A, accur, 0.5, feasible_solution, count_of_all_variables)
with_09 = InterPoint(C, A, accur, 0.9, feasible_solution, count_of_all_variables)
final_05 = multiply_and_sum_vectors(with_05, C)
final_09 = multiply_and_sum_vectors(with_09, C)

print("==========SECTION FOR COMPARING RESULTS==========")
print("Final answer from simplex", simplex_ans)
print("Final answer with alpha = 0.5:", with_05, "value of function:", final_05)
print("Final answer with alpha = 0.9:", with_09, "value of function:", final_09)
