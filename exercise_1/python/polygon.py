import sys

def read_input():
    lines = sys.stdin.read().splitlines()
    n = int(lines[0])
    points = [tuple(map(int, line.split())) for line in lines[1:n+1]]
    return points

# https://en.wikipedia.org/wiki/Shoelace_formula#Trapezoid_formula
def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n] # to loop back to the first point
        area += (x0 * y1 - x1 * y0)
    return abs(area) / 2

if __name__ == "__main__":
    points = read_input()
    print(polygon_area(points))
