import sys

"""
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

def read_decimal(num, precision=2):
    # read in decimal number and convert to integer
    sign = 1
    if num.startswith('-'):
        sign = -1
        num = num[1:]
    value = num.split(".")
    if len(value) == 1:
        value.append("0")
        
    whole = value[0]
    fractional = value[1][:precision]
    fractional = fractional + "0" * (precision - len(fractional))
    
    return sign * int(whole + fractional)

precision = 2

# read in number of points
n = int(sys.stdin.readline().strip())

gridsize = read_decimal(sys.stdin.readline().strip(), precision)

for _ in range(n):
    line = sys.stdin.readline().strip().split(" ")
    x = read_decimal(line[0], precision)
    y = read_decimal(line[1], precision)
    # snap to grid
    x = ((x // gridsize) * gridsize) / 10**precision
    y = ((y // gridsize) * gridsize) / 10**precision
    print(f"{x:.2f} {y:.2f}")