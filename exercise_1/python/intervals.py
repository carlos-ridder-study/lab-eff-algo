## Interval –  Efficient‑Algorithms Lab, Problem Set 1
import sys

def max_deletions(left_bound: int, right_bound: int, k: int) -> int:
    elements_in_interval = right_bound - left_bound + 1
    if k == 1:               # every element can delete itself
        return elements_in_interval

    ak = left_bound * k # check if first root can even be extended to chain
    if ak > right_bound:               # chunk 2 is empty –> all numbers are roots
        return 0

    chunk1_size = ak - left_bound                 # [a .. ak-1] , all numbers here are roots
    chunk2_size = right_bound - ak + 1             # [ak .. b] , only potentially root if not aleady a multiple of a number in chunk1

    # multiples of k in [ak .. b]
    multiples_in_chunk2 = (right_bound // k)  - ((ak - 1) // k)

    roots = chunk1_size + (chunk2_size - multiples_in_chunk2)
    return elements_in_interval - roots                     # deletable = total − roots


data = sys.stdin.read().strip().split()
a, b, k = map(int, data)
print(max_deletions(a, b, k))
