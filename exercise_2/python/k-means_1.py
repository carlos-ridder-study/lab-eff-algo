import sys

""""
# for debugging with input file
import os
input_file = os.environ.get('INPUT_FILE')
if input_file:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the input file
    input_path = os.path.join(script_dir, input_file)
    sys.stdin = open(input_path, 'r')
"""

line = sys.stdin.readline().strip().split(" ")
c = int(line[0])
q = int(line[1])
R = float(line[2])

# read in the centers
center = []
for _ in range(c):
    line = sys.stdin.readline().strip()
    center.append(float(line))

# sort the centers
center.sort()

# read in the points
points = []
for _ in range(q):
    line = sys.stdin.readline().strip()
    points.append(float(line))
    
for point in points:
    # find the closest center using binary search
    if point - center[-1] > R or point - center[0] < -R:
        print("none in range")
    else:
        low = 0   
        high = c - 1
        while low < high:
            mid = (low + high) // 2
            if point - center[mid] > 0:
                low = mid + 1
            else:
                high = mid
        min_dist = R + 1
        # closest center must lie in [low-1, low, low+1]
        closest = low
        for i in [-1, 0, 1]:
            if 0 <= low+i < c:
                if abs(point - center[low+i]) < min_dist:
                    min_dist = abs(point - center[low+i])
                    closest = low+i
        if min_dist <= R:
            print(center[closest])
        else:
            print("none in range")