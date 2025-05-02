# Input: The First line of the input contains the number n of vertices. Then, n lines follow describing the vertices in clockwise order along the boundary of the polygon. Each line contains the x and y coordinate of one vertex. You may assume that all coordinates are integers, and that we have −1000 ≤ x, y ≤ 1000.
# Output: Output the area A of the described polygon.

import sys
import math

# number of vertices - 1
n = int(sys.stdin.readline()) - 1

# we need the first coordinates again later
x0,y0 = sys.stdin.readline().split()
x_prev, y_prev = int(x0), int(y0)
area = 0

# for n-1 vertices: given -1000 <= x, y <= 1000, we take x = -1000 as
# the lower line to base the calculations on:
# calculate the average height of two adjacent vertices and multiply with the width.
# This value is positive if x_i-1 <= x_i and negative if x_i-1 > x_i
while n > 0:
    x_i, y_i = sys.stdin.readline().split()
    x = int(x_i)
    y = int(y_i)

    width = x - x_prev
    height = (1000 + y_prev + 1000 + y) / 2
    area = area + (width * height)

    # set the current vertex as the previous vertex
    x_prev = x
    y_prev = y
    n -= 1

# calculate the last step from vertex n to 1 like before
width = int(x0) - x_prev
height = (1000 + y_prev + 1000 + int(y0)) / 2
area = area + (width * height)

sys.stdout.write(str(area))