import sys

# read in matrix size
n = int(sys.stdin.readline().strip())

# read in # non-zero elements
m = int(sys.stdin.readline().strip())

# read in matrix elements
matrix = []
for _ in range(m):
    line = sys.stdin.readline().strip()
    value = line.split(" ")
    row = int(value[0])
    col = int(value[1])
    val = int(value[2])
    matrix.append((row, col, val))
    
# read in # non-zero elements in vector
b = int(sys.stdin.readline().strip())

# read in vector elements
vector = {}
for _ in range(b):
    line = sys.stdin.readline().strip()
    value = line.split(" ")
    row = int(value[0])
    val = int(value[1])
    vector[row] = val
    
# multiply matrix by vector
result = []
row = 0
sum = 0
for mrow, mcol, mval in matrix:
    if mrow == row:
        if mcol in vector:
            sum += mval * vector[mcol]
    elif mrow > row:
        if sum != 0:
            result.append((row, sum))
        row = mrow
        sum = 0
        if mcol in vector:
            sum += mval * vector[mcol]
if sum != 0:
    result.append((row, sum))
            
# output the result
for row, val in result:
    print(f"{row} {val}")
        

    
    

