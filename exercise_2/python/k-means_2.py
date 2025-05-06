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
    
class CoverTree:
    def __init__(self, point, level):
        self.point = point
        self.level = level
        self.children = []
        
    def insert(self, point, level):
        # Check if this node can cover the point at current level
        if (self.point - point).abs() > 2**level:
            return False

        for child in self.children:
            if child.insert(point, level - 1):
                return True

        # Insert as new child
        new_child = CoverTree(point, level - 1)
        self.children.append(new_child)
        return True

    def insert_point(self, point):
        level = self.level
        while not self.insert(point, level):
            # Promote root
            new_root = CoverTree(self.point, level + 1)
            new_root.children.append(self)
            old_point, old_children, old_level = self.point, self.children, self.level
            self.point = old_point
            self.level += 1
            self.children = [CoverTree(old_point, old_level)] + old_children
            level += 1
        
    def is_close(self, point, radius):
        # Check if a point is close to the cover tree node
        dist = (self.point - point).abs()
        if dist <= radius:
            return True
        for child in self.children:
            # use triangle inequality as lower bound of distance from point to child
            lower_bound = (child.point - point).abs() - 2**child.level
            if lower_bound <= radius:
                # children are close enough to check
                return child.is_close(point, radius)
        return False

line = sys.stdin.readline().strip().split(" ")
d = int(line[0])
R = float(line[1])

line = sys.stdin.readline().strip()
centers = []
R2 = R**2

"""
full linear search
while line != "":
    point = Vector([float(x) for x in line.split(" ")])
    dist = R2 + 1
    i = 0
    while dist > R2 and i < len(centers):
        dist = (point - centers[i]).squarednorm()
        i += 1
    if dist > R2:
        centers.append(point)
    line = sys.stdin.readline().strip()
        
print(len(centers))
"""
center_count = 0
if line != "":
    point = Vector([float(x) for x in line.split(" ")])
    center_count += 1
    cover_radius = math.ceil(math.log2(R))
    centers = CoverTree(point, cover_radius + 10)
    line = sys.stdin.readline().strip()

while line != "":
    point = Vector([float(x) for x in line.split(" ")])
    if not centers.is_close(point, R):
        center_count += 1
        centers.insert_point(point)
    line = sys.stdin.readline().strip()
    
print(center_count)