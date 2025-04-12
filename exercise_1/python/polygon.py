import sys

# read in number of points
n = int(sys.stdin.readline().strip())

area = 0.0

if n > 2:
    last_point = []
    first_point = []
    for _ in range(n):
        line = sys.stdin.readline().strip().split()
        x, y = float(line[0]), float(line[1])
        # Calculate area using shoelace formula
        if len(last_point) > 0:
            area += last_point[0] * y
            area -= x * last_point[1]
        else:
            first_point = [x, y]
        last_point = [x, y]
    area += last_point[0] * first_point[1]
    area -= first_point[0] * last_point[1]

print(abs(area) / 2.0)