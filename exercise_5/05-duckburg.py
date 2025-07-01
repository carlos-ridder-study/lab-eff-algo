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
i = int(sys.stdin.readline().strip())
n = int(sys.stdin.readline().strip())
S = [int(sys.stdin.readline().strip()) for _ in range(n)]

# algorithm goes here
def one_dkmeans(S, k):
    if len(S) <= k:
        return 0 
    S = sorted(S)
    n = len(S)
    C = [[0]*n for _ in range(n)]       # C[i][j] optimal cost of covering {x_i, ..., x_j} with a single center
    # Precompute prefix sums for fast range sum queries
    prefix = [0]*(n+1)
    for idx in range(n):
        prefix[idx+1] = prefix[idx] + S[idx]
    # Precompute all median costs in O(n^2)
    for l in range(n):
        for r in range(l, n):
            m = (l + r) // 2
            median = S[m]
            # Cost is sum of absolute deviations from the median
            # Use prefix sums for left and right parts
            left_count = m - l
            left_sum = prefix[m] - prefix[l]
            right_count = r - m
            right_sum = prefix[r+1] - prefix[m+1]
            C[l][r] = (median * left_count - left_sum) + (right_sum - median * right_count)
    
    dp = [[n*k]*k for _ in range(n)]    # dp[i][j] store optimal cost of covering {x_0, ..., x_j} with i+1 centers
    for i in range(n):
        dp[0][i] = C(0, i)
    for k_ in range(k-1)+1:
        for i in range(n):
            for j in range(i+1):
                dp[k_][i] = min(dp[k_][i], dp[k_-1][j] + C(j+1, i))
    return dp[k-1][n-1]

# output the result
result = one_dkmeans(S, i)
print(n*25-result)