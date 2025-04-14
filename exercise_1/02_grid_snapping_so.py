# Input: The first line contains the number of points n, which is at most 100. The second line contains the grid width with two digits after the decimal point. The coordinates are between −10000 and 10000. The following n lines each contain a point, consisting of two floating point numbers with two digits after the decimal point, separated by a space.
# Output: A line for each input point, containing the lower left corner of the grid cell that it falls into, with two digits after the decimal point.

import sys
import math
from decimal import *

# number of points n, at most 100
n = int(sys.stdin.readline())

# grid width with two digits after the decimal point
gridWidth = Decimal(sys.stdin.readline()).quantize(Decimal('1.00'))

# for n lines: output the lower left corner of the grid cell that each point falls into
while n > 0:
    value0, value1 = sys.stdin.readline().split()
    decimal0 = Decimal(value0).quantize(Decimal('1.00'))
    decimal1 = Decimal(value1).quantize(Decimal('1.00'))

    for x in range(2):
        if x == 0:
            num = decimal0
        else:
            sys.stdout.write(" ")  # formatting between two coordinates
            num = decimal1
        # if num is on grid, just print num
        if abs(num % gridWidth) == 0:
            sys.stdout.write(str(num.quantize(Decimal('1.00'))))
        else:
            factor = math.floor(num / gridWidth)
            sys.stdout.write(str((factor * gridWidth).quantize(Decimal('1.00'))))

    n = n - 1

    # newline if input is not finished
    if n > 0:
        print()
