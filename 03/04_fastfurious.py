import heapq
import sys
from collections import defaultdict


def calculate_manhattan_distance(first_coordinate, second_coordinate):
    x1, y1 = first_coordinate
    x2, y2 = second_coordinate
    return abs(x1 - x2) + abs(y1 - y2)


def prim_mst(points):
    n = len(points)
    adjacency_matrix = [[calculate_manhattan_distance(points[i], points[j]) for j in range(n)] for i in range(n)]

    visited = [False] * n
    #min_heap = [(0, 0, -1)]
    min_heap = [(0, 0)]  # (cost, vertex)
    mst_weight = 0
    #mst_edges = []

    while min_heap:
        #cost, vertex, parent = heapq.heappop(min_heap)
        cost, vertex = heapq.heappop(min_heap)
        if visited[vertex]:
            continue
        visited[vertex] = True
        mst_weight += cost
        if sum(visited) == len(visited):
            return mst_weight
        #if parent != -1:
        #    mst_edges.append((parent, vertex))

        for neighbor in range(n):
            if not visited[neighbor]:
                #heapq.heappush(min_heap, (adjacency_matrix[vertex][neighbor], neighbor, vertex))
                heapq.heappush(min_heap, (adjacency_matrix[vertex][neighbor], neighbor))

    return mst_weight  #, mst_edges


def construct_eulerian_tour(edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    stack = [next(iter(graph))]
    path = []

    while stack:
        node = stack[-1]
        if graph[node]:
            neighbor = graph[node].pop()
            graph[neighbor].remove(node)
            stack.append(neighbor)
        else:
            path.append(stack.pop())

    return path


n = int(sys.stdin.readline().strip())
checkpoints = []
for _ in range(n):
    x, y = map(int, sys.stdin.readline().strip().split())
    checkpoints.append((x, y))

# Calculate MST using Prim's Algorithm
cost_mst = prim_mst(checkpoints)
#tour = construct_eulerian_tour(edges)
#tour.append(tour[0])  # return to start

#cost = 0
#previous_checkpoint = checkpoints[tour[0]]
#for e in tour:
#    cost += calculate_manhattan_distance(previous_checkpoint, checkpoints[e])
#    previous_checkpoint = checkpoints[e]

#print(cost)
print(cost_mst * 2)
# Removing one edge from OPT gives a spanning tree (not minimal).
# Double the MST is a valid solution and is at most 2*OPT
