import sys
import re

"""
Example of input:
5 4
6 4|1 2|-1 1|0 1
24 6 1 2
"""


def greeting():
    print("""The input contains:
• A vector of coefficients of the objective function - C.
• A matrix of coefficients of the constraint function - A.
• A vector of right-hand side numbers - b.
• The approximation accuracy ϵ
============================
vector C input format:
x1 x2 x3... xn
for example:
5 4
============================
matrix A input format:
x1 x2|x1 x2|x1 x2|x1 x2
for example:
6 4|1 2|-1 1|0 1
============================
A vector of right-hand side numbers b input format:
sol1 sol2 sol3 sol3
for example:
24 6 1 2
============================
Output format:
• The string ”The method is not applicable!”
or
• A vector of decision variables - x∗.
• Maximum (minimum) value of the objective function
============================
""")


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

    print("Write A matrix in requirements format\n")
    A = input_matrix("")

    print("Write b vector in requirements format\n")
    b = input_vector("")

    accur = input_vector("Write the accuracy of the calculations: ", is_integer=True)
    print("============================")
    print("C vector:", C)
    print("Matrix A:", A)
    print("b vector:", b)
    return C, A, b, accur, len(C), len(A[0]), len(A)


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
    # print("============================")
    # print(S)
    # print(var_rows)
    return S, var_rows


greeting()  # function with rules

C, A, b, accur, count_of_all_variables, count_of_x_variables, count_of_equations = inp()  # input data from console

# check method applicable or not
check_matrix_size(C, A, b)
check_infeasibility(b)

# create titles for table
var_rows, var_colums = create_titles(count_of_x_variables, count_of_equations)

# create table S
S = [[]]

# fill z-row
for m in range(count_of_all_variables):
    if m < count_of_x_variables:
        S[0].append(-1 * C[m])
    else:
        S[0].append(0)

S[0].append(0)

# fill other rows
for n in range(count_of_equations):
    def_row = []
    for m in range(count_of_all_variables):
        if m < count_of_x_variables:
            def_row.append(A[n][m])
        else:
            def_row.append(1 if m == n + count_of_x_variables else 0)
    def_row.append(b[n])
    S.append(def_row)

# loop of iterations
while check_matrix(S):
    S, var_rows = Iteration(S, var_colums, var_rows)

print("============================")
# output data
for variable_index in range(len(var_rows)):
    variable = var_rows[variable_index]
    if "x" in variable:
        print(variable, format(S[variable_index][-1], f".{accur}f"))
for variable_index in range(len(var_rows)):
    variable = var_rows[variable_index]
    if variable == "z":
        print("Solution:", format(S[variable_index][-1], f".{accur}f"))
        break

for i in range(1, count_of_x_variables + 1):
    if f"x{i}" not in var_rows:
        print(f"x{i}", format(0, f".{accur}f"))
