# Given n x n matrix A, and vector v of size n. Compute vector Av
# Av = a0,0 * v0 + a0,1 * v1 + a0,2 * v2
#      a1,0 * v0 + a1,1 * v1 + a1,2 * v2
#      a2,0 * v0 + a2,1 * v1 + a2,2 * v2

import sys

# size of matrix A
n = int(sys.stdin.readline())
# number of non-zero entries in A
m = int(sys.stdin.readline())
A = []
v = []

for x in range(m):
    i, j, value = sys.stdin.readline().split()
    A.append((int(i), int(j), int(value)))

# number of non-zero entries in v
b = int(sys.stdin.readline())

for x in range(b):
    i, value = sys.stdin.readline().split()
    v.append((int(i), int(value)))

# computations: output the values of Av that are different from zero.
# for each i in A do: make sum of a_ij * v_i
i = 0
output = 0
not_first = False
for x in A:
    if i != x[0] and output != 0:
        if not_first:
            sys.stdout.write('\n')
        sys.stdout.write(str(i))
        sys.stdout.write(' ')
        sys.stdout.write(str(output))
        not_first = True
        output = 0

    i = x[0]
    for y in v:
        if y[0] == x[1]:
            output += x[2] * y[1]

# also print the last one
if output != 0:
    sys.stdout.write('\n')
    sys.stdout.write(str(i))
    sys.stdout.write(' ')
    sys.stdout.write(str(output))
