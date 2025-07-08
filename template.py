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

# read in lines from standard input
num_lines = int(sys.stdin.readline().strip())
input = [int(sys.stdin.readline().strip()) for _ in range(num_lines)]

# algorithm goes here
result = []

# output the result
for x in result:
    print(x)