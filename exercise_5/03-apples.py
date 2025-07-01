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

line = sys.stdin.readline().strip()
while line != "0":
    # read in lines from standard input
    n = int(line)
    ca, a = map(int, sys.stdin.readline().strip().split())
    cb, b = map(int, sys.stdin.readline().strip().split())

    # algorithm goes here
    def extended_gcd(a, b):
        """Computes the gcd of a and b and coefficients x, y such that xa + yb = gcd(a, b).

        Args:
            a (int)
            b (int)
        
        Returns:
            tuple: (gcd, x, y)
        """
        (old_r, r) = (a, b)
        (old_s, s) = (1, 0)
        (old_t, t) = (0, 1)
        while r != 0:
            quotient = old_r // r
            (old_r, r) = (r, old_r - quotient * r)
            (old_s, s) = (s, old_s - quotient * s)
            (old_t, t) = (t, old_t - quotient * t)
        return (old_r, old_s, old_t)
        

    (g, x0, y0) = extended_gcd(a, b)
    
    if n % g == 0:
        lower_bound = -(n*x0 // b)
        upper_bound = n*y0 // a
        if lower_bound > upper_bound:
            print("failed")
        else:
            if (cb/b) < (ca/a):
                t = lower_bound
            else:
                t = upper_bound
            x = (n*x0 + b * t) // g
            y = (n*y0 - a * t) // g
            print(f"{x} {y}")
    else:
        print("failed")
    line = sys.stdin.readline().strip()