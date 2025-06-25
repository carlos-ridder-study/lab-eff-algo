import sys
import random

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

def main():
    s = int(sys.stdin.readline().strip())
    n = int(sys.stdin.readline().strip())
    paths = []
    min_points = n
    shortest_path = None
    for _ in range(s):
        line = sys.stdin.readline().strip().split()
        m = int(line[0])
        points = []
        for k in range(m):
            x, y, z = map(int, line[1+3*k:1+3*(k+1)])
            points.append((x, y, z))
        paths.append(points)
        if m < min_points: 
            min_points = m
            shortest_path = points
        
     # Compute worst-case distance from chosen P to all others
    P = shortest_path
    worst = 0
    for idx in range(s):
        Q = paths[idx]
        # extend Q if shorter than n
        d2 = dist2(P, Q)
        if d2 > worst:
            worst = d2

    # Output the squared distance (already squared)
    print(int(worst))

# squared Euclidean distance
def squared_dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b))))


# compute the discrete Frechet-like distance squared between two paths P, Q
def dist2(P, Q):
        nP, mQ = len(P), len(Q)
        prev = [float('inf')] * mQ
        prev[0] = squared_dist(P[0], Q[0])
        # first row
        for j in range(1, mQ):
            prev[j] = max(prev[j-1], squared_dist(P[0], Q[j]))
        # DP rows
        for i in range(1, nP):
            cur = [float('inf')] * mQ
            cur[0] = max(prev[0], squared_dist(P[i], Q[0]))
            for j in range(1, mQ):
                d2 = squared_dist(P[i], Q[j])
                # transition from three possible predecessors
                best_prev = min(
                    max(prev[j], d2),      # up
                    max(cur[j-1], d2),     # left
                    max(prev[j-1], d2)     # diagonal
                )
                cur[j] = best_prev
            prev = cur
        return prev[mQ-1]


    
if __name__ == '__main__':
    main()