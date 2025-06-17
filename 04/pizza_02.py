import sys
import random


def remove_edge(edges, node):
    return [e for e in edges if node not in e]


def vertex_cover(edges, k):
    if not edges:
        return True

    if k == 0:
        return False

    # pick random edge
    u, v = random.choice(edges)

    if vertex_cover(remove_edge(edges, u), k - 1):
        return True

    if vertex_cover(remove_edge(edges, v), k - 1):
        return True

    return False


n, m = map(int, sys.stdin.readline().strip().split())
edges = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(m)]
if vertex_cover(edges, 6):
    print("possible")
else: print("impossible")