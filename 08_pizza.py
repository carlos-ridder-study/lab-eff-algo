import sys
import random

edges = []
n, m = map(int, sys.stdin.readline().split())

for _ in range(m):
    u, v = map(int, sys.stdin.readline().split())
    edges.append((u, v))

cover = []
while len(edges) > 0:
    i = random.randrange(len(edges))
    (u, v) = edges.pop(i) # pick any {u,v}
    cover.append((u, v))
    # delete all edges incident to either u or v
    edges = [edge for edge in edges if u not in edge and v not in edge]

vertices = [vertices for pair in cover for vertices in pair]
sys.stdout.write(str(len(vertices)))
