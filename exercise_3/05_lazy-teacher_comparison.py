
import random
import time
import sys
import math
from collections import defaultdict 
from itertools import combinations
from copy import deepcopy  

def convolve_dp(dp, p):
    """Add one voter with probability p to existing distribution dp."""
    next_dp = defaultdict(float)
    for s in dp:
        next_dp[s]     += dp[s] * (1 - p)
        next_dp[s + 1] += dp[s] * p
    return next_dp

def greedy_pairs_dp(p_list, k): 
    n = len(p_list)  
    candidates_set = set(range(n))
    dp = defaultdict(float)
    dp[0] = 1.0  # Start with 0 voters, 0 yes votes

    # hyperparameters
    low_count = min(30, n // 2)
    high_count = min(30, n // 2)
    balanced_count = min(30, n)

    for step in range(k):  # Select k pairs
        best_pair = None
        best_prob = -1.0
        best_new_dp = None
        
        """
        # Build a sorted list of current candidate indices by p value.
        current_candidates = sorted(list(candidates_set), key=lambda i: p_list[i])
        # Extreme low candidates and high candidates come from the ends.
        low_group = current_candidates[:low_count]
        high_group = current_candidates[-high_count:]
        # Balanced candidates: sorted by distance to 0.5.
        balanced_group = sorted(current_candidates, key=lambda i: abs(p_list[i] - 0.5))[:balanced_count]

        # Form permitted pair candidates: cross pairs from low and high, plus pairs from balanced.
        candidate_pairs = set()
        for i in low_group:
            for j in high_group:
                # Ensure order so each pair is unique.
                if i < j:
                    candidate_pairs.add((i,j))
                else:
                    candidate_pairs.add((j,i))
        for pair in combinations(balanced_group, 2):
            candidate_pairs.add(pair)
        """
        candidate_pairs = set(combinations(candidates_set, 2))
        
        # Precompute convolved dp for each candidate in current dp for reuse.
        conv_cache = {}

        for i, j in candidate_pairs:
            # Try adding (i, j)
            if i not in conv_cache:
                dp_i = convolve_dp(dp, p_list[i])
                conv_cache[i] = dp_i
            dp_i = conv_cache[i]
            dp_ij = convolve_dp(dp_i, p_list[j])
            tie_prob = dp_ij[(2 * (step + 1)) // 2]

            if tie_prob > best_prob:
                best_prob = tie_prob
                best_pair = (i, j)
                best_new_dp = dp_ij

        if best_pair is None:
            break

        i, j = best_pair
        candidates_set.remove(i)
        candidates_set.remove(j)
        dp = best_new_dp  # Update current distribution

    return dp[k]


def tie_probability(probs):
    """ Compute tie probability using dynamic programming."""
    dp = defaultdict(float)
    dp[0] = 1.0
    for p in probs:
        next_dp = defaultdict(float)
        for s in dp:
            next_dp[s]     += dp[s] * (1 - p)
            next_dp[s + 1] += dp[s] * p
        dp = next_dp
    n = len(probs)
    return dp[n // 2] if n % 2 == 0 else 0.0

def brute_force_optimal(p_list, k):
    """Try all subsets of 2k voters and return the one with the highest tie probability."""
    n = len(p_list)
    best_subset = None
    best_prob = -1.0
    for subset in combinations(range(n), 2 * k):
        probs = [p_list[i] for i in subset]
        prob = tie_probability(probs)
        if prob > best_prob:
            best_prob = prob
            best_subset = subset
    return list(best_subset), best_prob

def max_tie_probability(p_list, k):
    n = len(p_list)
    dp = [[[-1.0 for _ in range(k+1)] for _ in range(2*k+1)] for _ in range(n+1)]
    dp[0][0][0] = 1.0
    for i in range(n):
        for s in range(0, min(i, 2*k)+1):
            for t in range(0, min(s, k)+1):
                val = dp[i][s][t]
                p = p_list[i]
                if val < 0:
                    continue
                if dp[i + 1][s][t] < val:
                    dp[i + 1][s][t] = val
                if s + 1 <= 2*k:
                    # vote yes
                    if t + 1 <= k:
                        new_val = p*val
                        if dp[i+1][s+1][t+1] < new_val:
                            dp[i + 1][s+1][t+1] = new_val
                    
                    # vote no
                    new_val = (1-p)*val
                    if dp[i + 1][s+1][t] < new_val:
                        dp[i + 1][s+1][t] = new_val
    max_tie = dp[n][2*k][k]
    if max_tie < 0:
        return 0.0
    return max_tie

def solve_lazy_teacher_optimal_subset(probabilities, k):
    """
    Solves the Lazy Teacher problem using dynamic programming to find the maximum
    probability of a tie over all unique subsets of 2*k students.
    """


    # dp[j][l] will store the maximum probability of selecting exactly j students
    # from the first 'i' students and getting exactly l 'increase' votes.
    # We use a 2D array of size (k+1) x (k+1) for space optimization,
    # where the first dimension represents the number of students selected (j)
    # and the second dimension represents the number of increase votes (l).
    # Initialize with -1.0 to represent unreachable states, as probabilities are non-negative.
    dp = [[0.0 for _ in range(k + 1)] for _ in range(2*k + 1)]

    # Base case: selecting 0 students from the first 0 students,
    # getting 0 increase votes has a maximum probability of 1.0.
    dp[0][0] = 1.0

    # Iterate through each student (from 1 to n)
    for p in probabilities:
        # Iterate backwards through the number of students selected so far (j).
        # We go from k down to 1 because to form a committee of size j including
        # the current student, we must have formed a committee of size j-1
        # from the previously considered students.
        for j in range(2*k, 0, -1):
            # Iterate backwards through the number of increase votes (l).
            # The maximum number of increase votes cannot exceed the number of students selected (j).
            for l in range(min(k, j), -1, -1):
                # Maximum probability of reaching state (j, l) after considering the current student:
                # This state can be reached in two main ways involving the current student (if included):

                # Case 1: Include the current student, and they vote to increase (with probability p).
                # To get l increase votes with j students including the current one,
                # we needed l-1 increase votes from j-1 students before.
                prob_yes = 0.0
                if l > 0 and dp[j - 1][l - 1] > 0: # Check if previous state was reachable
                    prob_yes = dp[j - 1][l - 1] * p

                # Case 2: Include the current student, and they vote to decrease (with probability 1-p).
                # To get l increase votes with j students including the current one,
                # we needed l increase votes from j-1 students before.
                prob_no = 0.0
                if dp[j - 1][l] > 0: # Check if previous state was reachable
                    prob_no = dp[j - 1][l] * (1 - p)

                # The new maximum probability for state (j, l) is the maximum of:
                # - The maximum probability of achieving state (j, l) without including the current student (old dp[j][l]).
                # - The maximum probability of achieving state (j, l) by including the current student (max(prob_if_increase, prob_if_decrease)).
                dp[j][l] = max(dp[j][l], prob_yes + prob_no)

    # The problem asks for the maximum probability of a tie, which occurs
    # when we have selected exactly k students and obtained exactly k/2
    # increase votes. This probability is stored in dp[k][k // 2] after
    # processing all n students.
    return dp[2*k][k]


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
    
    random.shuffle(best_subset)
    random.shuffle(rest)

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




"""
n, k = map(int, sys.stdin.readline().split())
k = k // 2
p_list = []
line = sys.stdin.readline().strip()
while line:
    p_list.append(float(line))
    line = sys.stdin.readline().strip()

#max_tie = greedy_pairs_dp(p_list, k)
# print(round(max_tie, 3))
"""


N_RUNS = 5
N = [30, 60, 100]
K = [7, 16, 24]

# We will accumulate total relative errors and times.
total_time_local_search = 2*[0]
total_time_local_search_iterative = 2*[0]
total_time_hill_climber = 2*[0]
total_time_greedy = 0
total_time_dp_subsets = 0
total_time_dp_dynamic = 0
total_time_upper_bound = 0

total_err_local_search = 0
total_correct_local_search = 0
total_err_local_search_iterative = 2*[0]
total_correct_local_search_iterative = 2*[0]
total_error_hill_climber = 2*[0]
total_err_greedy = 0
total_correct_greedy = 0
total_err_dp_subsets = 0
total_err_dp_dynamic = 0
total_err_upper_bound = 0

valid_runs = 0
random_start = 0

for n, k in zip(N, K):
    for _ in range(2*N_RUNS):
        random_start = (random_start+1)%2
        # Generate a random p_list of n probabilities.
        p_list = [random.uniform(0, 1) for _ in range(n)]
        
        # Compute the optimal (brute-force) solution on the current instance.
        # Note: brute_force_optimal returns (selected_subset, optimal_prob)
        # selected_opt, prob_opt = brute_force_optimal(p_list, k)
        
        # Only process if we have a positive optimal probability.
        #if prob_opt <= 0:
        #    continue
        
        # valid_runs += 1
        valid_runs = N_RUNS
        
                # Local search:
        t_start = time.time()
        if random_start == 1:
            selected = random.sample(range(n), 2 * k)
            tie_prob = compute_tie_probability(p_list, selected)
        else:
            tie_prob, selected = max_tie_probability_with_subsets(p_list, k)
        prob_local_search, _ = local_search(selected, tie_prob, p_list, k)
        time_local_search = time.time() - t_start
        total_time_local_search[random_start] += time_local_search

        
        
        
        # DP with subsets:
        t_start = time.time()
        prob_dp_subsets, selected = max_tie_probability_with_subsets(p_list, k)
        total_time_dp_subsets += time.time() - t_start
        
        """
        # Hill climber:
        t_start = time.time()
        if random_start == 1:
            selected = random.sample(range(n), 2 * k)
            tie_prob = compute_tie_probability(p_list, selected)
        else:
            tie_prob, selected = max_tie_probability_with_subsets(p_list, k)
        prob_hill_climber = hill_climber(p_list, selected)
        total_time_hill_climber[random_start] += time.time() - t_start
        
        # local search iterative:
        t_start = time.time()
        if random_start == 1:
            selected = random.sample(range(n), 2 * k)
            tie_prob = compute_tie_probability([p_list[i] for i in selected], k)
        else:
            tie_prob, selected = max_tie_probability_with_subsets(p_list, k)
        prob_local_search_iterative, _ = local_search_iterative(selected, tie_prob, p_list, k)
        total_time_local_search_iterative[random_start] += time.time() - t_start


        # Greedy pairs DP:
        t_start = time.time()
        prob_greedy = greedy_pairs_dp(p_list, k)
        time_greedy = time.time() - t_start
        
        # Dynamic Programming:
        t_start = time.time()
        prob_dp_dynamic = max_tie_probability(p_list, k)
        time_dp_dynamic = time.time() - t_start
        
        # DP Upper Bound:
        t_start = time.time()
        prob_upper_bound = solve_lazy_teacher_optimal_subset(p_list, k)
        time_upper_bound = time.time() - t_start


        # Compute relative error for each; relative error = (optimal - algorithm_result) / optimal.
        err_local_search = (prob_opt - prob_local_search) / prob_opt
        err_local_search_iterative = (prob_opt - prob_local_search_iterative) / prob_opt
        err_greedy = (prob_opt - prob_greedy) / prob_opt
        err_dp_subsets = (prob_opt - prob_dp_subsets) / prob_opt
        err_dp_dynamic = (prob_opt - prob_dp_dynamic) / prob_opt
        err_upper_bound = (prob_upper_bound - prob_opt) / prob_upper_bound  # upper bound should be >= optimal
        """
        #total_time_greedy += time_greedy
        #total_time_dp_subsets += time_dp_subsets
        #total_time_dp_dynamic += time_dp_dynamic
        #total_time_upper_bound += time_upper_bound
        
        """
        total_err_local_search += err_local_search
        if round(prob_opt - prob_local_search, 3) == 0:
            total_correct_local_search += 1
        total_err_local_search_iterative += err_local_search_iterative
        if round(prob_opt - prob_local_search_iterative, 3) == 0:
            total_correct_local_search_iterative += 1
        total_err_greedy += err_greedy
        if round(prob_opt - prob_greedy, 3) == 0:
            total_correct_greedy += 1
        total_err_dp_subsets += err_dp_subsets
        total_err_dp_dynamic += err_dp_dynamic
        total_err_upper_bound += err_upper_bound
        """

    # Compute averages if at least one valid run occurred.
    if valid_runs > 0:

        """
        avg_err_local_search = total_err_local_search / valid_runs
        avg_err_local_search_iterative = total_err_local_search_iterative / valid_runs
        avg_err_greedy = total_err_greedy / valid_runs
        avg_err_dp_subsets = total_err_dp_subsets / valid_runs
        avg_err_dp_dynamic = total_err_dp_dynamic / valid_runs
        avg_err_upper_bound = total_err_upper_bound / valid_runs
        """

        """
        print("Averages over", valid_runs, "runs:")
        print("[Local search iterative]: Probability of correct solution = {:.3f} Average relative error = {:.3f}, Average time = {:.6f} sec".format(total_correct_local_search_iterative/valid_runs, avg_err_local_search_iterative, avg_time_local_search_iterative))
        print("[Local search]: Probability of correct solution = {:.3f} Average relative error = {:.3f}, Average time = {:.6f} sec".format(total_correct_local_search/valid_runs, avg_err_local_search, avg_time_local_search))
        print("[Greedy pairs DP]: Probability of correct solution = {:.3f} Average relative error = {:.3f}, Average time = {:.6f} sec".format(total_correct_greedy/valid_runs, avg_err_greedy, avg_time_greedy))
        print("[DP with subsets]: Average relative error = {:.3f}, Average time = {:.6f} sec".format(avg_err_dp_subsets, avg_time_dp_subsets))
        print("[Dynamic Programming]: Average relative error = {:.3f}, Average time = {:.6f} sec".format(avg_err_dp_dynamic, avg_time_dp_dynamic))
        print("[DP Upper Bound]: Average relative error = {:.3f}, Average time = {:.6f} sec".format(avg_err_upper_bound, avg_time_upper_bound))
        """
        print("Averages over", valid_runs, "runs for n =", n, "k =", k, "with optimzed start:")
        print("[Local search recursive]: Average time = {:.6f} sec".format(total_time_local_search[0] / valid_runs))
        #print("[Hill climber]: Average time = {:.6f} sec".format(total_time_hill_climber[0] / valid_runs))
        #print("[Local search iterative]: Average time = {:.6f} sec".format(total_time_local_search_iterative[0] / valid_runs))
        print("with random start:")
        print("[Local search recursive]: Average time = {:.6f} sec".format(total_time_local_search[1] / valid_runs))
        #print("[Hill climber]: Average time = {:.6f} sec".format(total_time_hill_climber[1] / valid_runs))
        #print("[Local search iterative]: Average time = {:.6f} sec".format(total_time_local_search_iterative[1] / valid_runs))
        print("[DP with subsets]: Average time = {:.6f} sec".format(100*(total_time_dp_subsets / valid_runs)))
        print("\n")        
    else:
        print("No valid runs with positive optimal probability were obtained.")