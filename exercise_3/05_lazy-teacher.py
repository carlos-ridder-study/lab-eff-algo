import sys
import random
import math
from copy import deepcopy  

import os
input_file = os.environ.get('INPUT_FILE')
if input_file:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the input file
    input_path = os.path.join(script_dir, input_file)
    sys.stdin = open(input_path, 'r')


def compute_tie_probability(probabilities, subset):
    """Compute the probability that exactly k out of 2k people vote yes."""
    k = len(subset)
    dp = [0.0] * (k + 1)
    dp[0] = 1.0
    for i in subset:
        p = probabilities[i]
        for s in range(k, 0, -1):
            dp[s] = dp[s] * (1 - p) + dp[s - 1] * p
        dp[0] *= (1 - p)
    return dp[k//2]

def max_tie_probability_with_subsets(probs, k):
    n = len(probs)
    # dp[i][s] = (placeholder_score, subset of s indices from first i people)
    dp = [[(0.0, []) for _ in range(2 * k + 1)] for _ in range(n + 1)]
    dp[0][0] = (1.0, [])

    for i in range(n):
        for s in range(min(i, 2 * k) + 1):
            prob, subset = dp[i][s]
            # Case 1: skip i
            if prob > dp[i + 1][s][0]:
                dp[i + 1][s] = (prob, deepcopy(subset))
            # Case 2: include i
            if s + 1 <= 2 * k:
                new_subset = subset + [i]
                if prob > dp[i + 1][s + 1][0]:
                    dp[i + 1][s + 1] = (prob, new_subset)

    # Final step: evaluate all complete subsets and compute tie probabilities
    best_prob = 0.0
    best_subset = []
    for i in range(n + 1):
        _, subset = dp[i][2 * k]
        if len(subset) == 2 * k:
            tie_prob = compute_tie_probability(probs, subset)
            if tie_prob > best_prob:
                best_prob = tie_prob
                best_subset = subset

    return best_prob, best_subset

def local_search(selected, tie_prob, probs, k, temperature=0):
    """
    Perform a local search to find the best subset of size 2*k with the maximum tie probability.
    """
    candidates = [i for i in range(len(probs)) if i not in selected]
    random.shuffle(candidates)
    best_subset = selected
    best_prob = tie_prob
    # Shuffle a copy of the selected list to iterate in random order)
    random.shuffle(selected)
    for i in range(len(selected)):
        for j in candidates:
            new_subset = selected.copy()
            new_subset[i] = j
            new_prob = compute_tie_probability(probs, new_subset)
            delta = new_prob - best_prob
            if delta > 0:
                return local_search(new_subset, new_prob, probs, k, 0.9*temperature)
            elif delta < 0 and temperature > 0:
                p = random.uniform(0, 1)
                if p < math.exp(delta / temperature):
                    return local_search(new_subset, new_prob, probs, k, 0.9*temperature)
    return best_prob, best_subset

def local_search_iterative(selected, tie_prob, probs, k, temperature=0): 
    """
    Iteratively perform local search to improve a subset of size 2*k with maximum tie probability. """ 
    best_subset = selected
    best_prob = tie_prob
    improved = True
    
    while improved:
        improved = False
        # List candidates not in the current best subset
        candidates = [i for i in range(len(probs)) if i not in best_subset]
        random.shuffle(candidates)
        # Iterate over indices of the selected subset
        for i in range(len(best_subset)):
            for j in candidates:
                new_subset = best_subset.copy()
                new_subset[i] = j
                new_prob = compute_tie_probability(probs, new_subset)
                delta = new_prob - best_prob
                if delta > 0:
                    best_subset = new_subset
                    best_prob = new_prob
                    improved = True
                    # Break to restart search from the new best state
                    break
            if improved:
                break
    return best_prob, best_subset

# We can use a hillclimbing algorithm to improve starting from some solution by swapping items.
def hill_climber(probabilities, subset):
    k = len(subset)
    best_subset = subset.copy()
    best_score = compute_tie_probability(probabilities, best_subset)

    # Elements are not unique
    def multiset_difference(a, b):
        a_copy = a.copy()
        for x in b:
            if x in a_copy:
                a_copy.remove(x)
        return a_copy

    rest = multiset_difference(list(range(n)), best_subset)

    # run until no local improvement can be found anymore
    improved = True
    while improved:
        improved = False
        for i in range(len(best_subset)):
            for j in range(len(rest)):
                new_subset = best_subset.copy()
                new_subset[i] = rest[j]
                new_score = compute_tie_probability(probabilities, new_subset)

                if new_score > best_score:
                    # Better search point found -> Accept swap
                    replaced = best_subset[i]
                    best_subset = new_subset
                    best_score = new_score

                    # Update rest list for next iteration
                    rest[j] = replaced
                    improved = True
                    break  # Restart inner loop after an improvement
            if improved:
                break  # Restart outer loop too

    return best_score # , best_subset

n, k = map(int, sys.stdin.readline().strip().split())
k = k // 2
p_list = []
line = sys.stdin.readline().strip()
while line:
    p_list.append(float(line))
    line = sys.stdin.readline().strip()

selected = random.sample(range(n), 2 * k)
tie_prob = compute_tie_probability(p_list, selected)
#for i in range(100):
#    random.shuffle(p_list)
#    tie_prob, selected = max_tie_probability_with_subsets(p_list, k)
#    if tie_prob > max_tie:
#        max_tie = tie_prob
max_tie, _ = local_search(selected, tie_prob, p_list, k)
print(round(max_tie, 3))