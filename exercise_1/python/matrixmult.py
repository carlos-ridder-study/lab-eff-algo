## Matrix Mult –  Efficient‑Algorithms Lab, Problem Set 1
import sys

## just get as list  and step through
data = list(map(int, sys.stdin.buffer.read().split()))
file_number_counter = 0

n = data[file_number_counter]        # matrix dimension (unused later)
file_number_counter += 1   
m = data[file_number_counter]        # non‑zeros in A
file_number_counter += 1   


# sparse matrix
A = []
for _ in range(m):
    i = data[file_number_counter]
    j = data[file_number_counter + 1]
    matrix_value = data[file_number_counter + 2]

    A.append((i, j, matrix_value))
    file_number_counter += 3

#  vector into a dict for fast
b = data[file_number_counter]; # non zeroes in v
file_number_counter += 1
v = {}

for _ in range(b):
    j = data[file_number_counter];   # index
    v[j] = data[file_number_counter + 1] #  value
    file_number_counter += 2

# multiply 
result_vector = {}
for i, j, A_ij in A: # step entries in the matrix and add onto result cell in new vector
    vector_entry_j = v.get(j)
    if vector_entry_j is not None: # skip if value is 0 or not even there
        result_vector[i] = result_vector.get(i, 0) + A_ij * vector_entry_j #add onto


# result values might be zero, so dont ouput them if so
for i in sorted(result_vector):
    if result_vector[i] != 0:
        sys.stdout.write(f"{i} {result_vector[i]}\n")
