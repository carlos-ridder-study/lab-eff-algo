import random
import sys
import time


# Calculate the probability of a tie for a set of probabilities with even cardinality.
def calculate_tie_probability(probabilities):
    k = len(probabilities)
    dp = [0.0] * (k + 1)
    dp[0] = 1.0

    for p in probabilities:
        next_dp = [0.0] * (k + 1)
        for i in range(k):
            next_dp[i] += dp[i] * (1 - p)
            next_dp[i + 1] += dp[i] * p
        dp = next_dp

    return dp[k // 2]


# We can use a hillclimbing algorithm to improve starting from some solution by swapping items.
def hill_climber(probabilities, subset):
    k = len(subset)
    best_subset = subset.copy()
    best_score = calculate_tie_probability(best_subset)

    # Elements are not unique
    def multiset_difference(a, b):
        a_copy = a.copy()
        for x in b:
            if x in a_copy:
                a_copy.remove(x)
        return a_copy

    rest = multiset_difference(probabilities, best_subset)

    # run until no local improvement can be found anymore
    improved = True
    while improved:
        improved = False
        for i in range(len(best_subset)):
            for j in range(len(rest)):
                new_subset = best_subset.copy()
                new_subset[i] = rest[j]
                new_score = calculate_tie_probability(new_subset)

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


n, k = map(int, sys.stdin.readline().split())

# setting n < 0 enables benchmarking
if n < 0:
    n = 25
    k = 12
    while True:
        probabilities = [round(random.uniform(0.0, 1.0), 3) for _ in range(n)]
        start_set = random.sample(probabilities, k)
        start_time = time.time()
        probability = hill_climber(probabilities, start_set)
        time_random = time.time() - start_time

        while True:
            start_set = random.sample(probabilities, k)
            mean = 0.0
            for p in start_set:
                mean += p
            if abs(mean - (k // 2)) < 0.5:  # this could potentially not terminate
                break

        start_time = time.time()
        probability_init = hill_climber(probabilities, start_set)
        time_init = time.time() - start_time

        if round(probability_init, 3) != round(probability, 3):
            print(probability_init, probability)
            print("Incorrect")
            break
        else:
            print(round(time_random - time_init, 5), time_init)

probabilities = [float(sys.stdin.readline()) for _ in range(n)]
start_set = []

while True:
    start_set = random.sample(probabilities, k)
    mean = 0.0
    for p in start_set:
        mean += p
    if abs(mean - (k // 2)) < 0.5:  # this could potentially not terminate
        break

probability = hill_climber(probabilities, start_set)
print(round(probability, 3))
