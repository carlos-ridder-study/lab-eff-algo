# Input: Starts with a line containing the number m of corners of the polygon. Then, m lines follow;
# where the i-th line contains the x- and y-coordinate of the i-th corner. The coordinates are all integers.
# Assume that polygon is non-self-intersecting, but more than 2 corners can be on a line.
#
# Output: Output the number of crossroads that are located in your district. You are also
# responsible for the crossroads that lie on the boundary.
import sys
import math

m = int(sys.stdin.readline().strip())
polygon = []

for _ in range(m):
    x, y = map(int, sys.stdin.readline().strip().split())
    polygon.append((x, y))


# Calculate number of points on the boundary
# In a Cartesian coordinate system, gcd(a, b) can be interpreted as the number of segments between points with
# integral coordinates on the straight line segment joining the points (0, 0) and (a, b).
def count_boundary_points(polygon):
    p = 0
    for i in range(m):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % m]
        p += math.gcd(abs(x2 - x1), abs(y2 - y1))
    return p


# calculate using the shoelace formula
def area_of_polygon(polygon):
    polygon_area = 0
    for i in range(m):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % m]
        polygon_area += x1 * y2
        polygon_area -= y1 * x2
    return abs(polygon_area/2)


# Pick's Theorem: Suppose that a polygon has integer coordinates for all of its vertices. Let i be the number
# of integer points interior to the polygon, and let b be the number of integer points on its boundary.
# Then the area A of this polygon is: A = i + (b/2) - 1
# So; i = A - (b/2) + 1
area = area_of_polygon(polygon)
boundary_points = count_boundary_points(polygon)
points_inside_boundary = math.floor(area - (boundary_points / 2) + 1)

sys.stdout.write(str(boundary_points + points_inside_boundary))
