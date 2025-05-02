# Input: consists of three integers a,b,k, where 10**11 >= a,b >= 0,k > 0.
# The numbers a and b describe the left and right bounds of the interval, i.e.,
# the set {a, a+1,…,b-1,b}
# Output: The output is a single integer r , which is the maximum
# amount of numbers you can delete when following the optimal strategy.

import sys
import math

a, b, k = sys.stdin.readline().split()
a = int(a)
b = int(b)
k = int(k)

# delete x from the interval if there exists a <= y <= b so that k * x = y
# smallest x: ceil(a / k) --> x*k <= b --> is (x+1)*k <= b
# --> upper bound: floor(b/k)
# output: floor(b/k) - ceil(a/k) + 1

sys.stdout.write(str(math.floor(b/k) - max(math.ceil(a/k), a) + 1))

# [0,1,2,3,4,5,6,7,8,9,10] , k=3
# floor(10/3) = 3
# ceil(0/3) = 0

# [1,2,3,4,5,6] , k=2
# floor(6/2) = 3
# ceil(1/2) = 1

# [3,4,5,6,7,8], k=2
# floor(8/2) = 4
# ceil(3/2) = 2
