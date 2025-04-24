import sys
import math

# for debugging with input file
import os
input_file = os.environ.get('INPUT_FILE')
if input_file:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the input file
    input_path = os.path.join(script_dir, input_file)
    sys.stdin = open(input_path, 'r')

class Vector:
    def __init__(self, coords):
        self.coords = coords
    
    def abs(self):
        return math.sqrt(sum(a ** 2 for a in self.coords))
    
    def squarednorm(self):
        return sum(a ** 2 for a in self.coords)
    
    def __add__(self, other):
        # Element-wise addition of points
        return Vector([a + b for a, b in zip(self.coords, other.coords)])
        
    def __sub__(self, other):
        # Element-wise subtraction of points
        return Vector([a - b for a, b in zip(self.coords, other.coords)])
    
    def __str__(self):
        return " ".join(str(x) for x in self.coords)

line = sys.stdin.readline().strip().split(" ")
d = int(line[0])
R = float(line[1])

line = sys.stdin.readline().strip()
centers = []
R = R**2

while line != "":
    line = line.split(" ")
    coords = []
    for i in range(d):
        coords.append(float(line[i]))
    point = Vector(coords)
    dist = R + 1
    i = 0
    while dist > R and i < len(centers):
        dist = (point - centers[i]).squarednorm()
        i += 1
    if dist > R:
        centers.append(point)
    line = sys.stdin.readline().strip()
        
print(len(centers))