import sys
import random
import math

booths = []
n, k = map(int, sys.stdin.readline().split())

for _ in range(n):
    x, y = map(int, sys.stdin.readline().split())
    booths.append((x,y))

extinguishers = [booths.pop(random.randrange(len(booths)))]


def calculate_distance(s, loc):
    distance = math.inf
    for ex in s:  # use the Euclidean distance
        distance = min(distance, math.sqrt((ex[0] - loc[0]) ** 2 + (ex[1] - loc[1]) ** 2))
    return distance


while len(extinguishers) < k:
    max_distance = 0
    max_index = 0
    for i in range(len(booths)):
        distance = calculate_distance(extinguishers, booths[i])
        if distance > max_distance:
            max_index = i
            max_distance = distance
    extinguishers.append(booths.pop(max_index))

# calculate max distance of booth to fire extinguisher
max_distance = 0
for booth in booths:
    max_distance = max(max_distance, calculate_distance(extinguishers, booth))

sys.stdout.write(str(max_distance ** 2))
