import sys

n = int(sys.stdin.readline())
house_index = dict()
last_index = 0
clauses = []

for i in range(n):
    line = sys.stdin.readline().split()
    parity = 1 if line[-1] == 'odd' else 0
    houses = line[:-1]

    row = []
    for house in houses:
        if house not in house_index:
            house_index[house] = last_index
            last_index += 1
        row.append(house_index[house])
    clauses.append((row, parity))

A = [[0]*last_index for i in range(len(clauses))]
b = [0]*len(clauses)

for i, (row, parity) in enumerate(clauses):
    for idx in row:
        A[i][idx] = 1
    b[i] = parity


# Gaussian elimination over GF(2). Assume linear system is solvable
def gauss(A, b):
    n = len(A)
    m = len(A[0])
    row = 0
    for col in range(m):
        pivot = -1
        for i in range(row, n):
            if A[i][col]:  # A[i][col] == 1
                pivot = i
                break
        if pivot == -1:
            continue
        A[row], A[pivot] = A[pivot], A[row]  # swap row and pivot
        b[row], b[pivot] = b[pivot], b[row]  # swap row and pivot
        for i in range(n):
            if i != row and A[i][col]:
                for j in range(m):
                    A[i][j] ^= A[row][j]  # A[i][j] = A[i][j] - A[row][j]
                b[i] ^= b[row]  # b[i] = b[i] - b[row]
        row += 1
    return b


solution = gauss(A, b)
print(sum(solution))
