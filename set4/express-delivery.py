import sys
import heapq
import math

def parse_input():
    input_lines = sys.stdin.read().splitlines()
    ptr = 0

    # First line: number of stations (n) and number of queries (q)
    n, q = map(int, input_lines[ptr].split())
    ptr += 1

    # Next n lines: capacity and speed of each station
    capacities = []
    speeds = []
    for _ in range(n):
        c, s = map(int, input_lines[ptr].split())
        capacities.append(c)
        speeds.append(s)
        ptr += 1

    # Next n lines: n x n distance matrix
    distances = []
    for _ in range(n):
        row = list(map(int, input_lines[ptr].split()))
        distances.append(row)
        ptr += 1

    # Next q lines: delivery queries
    queries = []
    for _ in range(q):
        u, v = map(int, input_lines[ptr].split())
        queries.append((u - 1, v - 1))  # adjust to 0-based indexing if needed
        ptr += 1

    return n, q, capacities, speeds, distances, queries


def floyd_warshall(n, distances):
    # init the shortest path matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if distances[i][j] != -1:
                dist[i][j] = distances[i][j]
        dist[i][i] = 0  # dist to self is 0

    # Floyd-Warshall core loop
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def build_delivery_network(n, apsp, speeds, capacities):
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if apsp[i][j] <= capacities[i]:
                time = apsp[i][j] / speeds[i]
                graph[i].append((j, time))  # edge from i → j
    return graph



def dijkstra(n, graph, src):
    print(type(graph))
    dist = [math.inf] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        cur_time, u = heapq.heappop(pq)
        if cur_time > dist[u]:
            continue
        for v, time in graph[u]:
            if dist[u] + time < dist[v]:
                dist[v] = dist[u] + time
                heapq.heappush(pq, (dist[v], v))
    return dist


if __name__ == "__main__":
    n, q, capacities, speeds, distances, queries = parse_input()

    print("Stations:", n)
    print("Queries:", q)
    print("Capacities:", capacities)
    print("Speeds:", speeds)
    print("Distances:")
    for row in distances:
        print(row)
    print("Queries:", queries)


    # shortest distance map, since we cant assume triangle inequality holds
    all_pairs_shortest_paths= floyd_warshall(n, distances=distances)
    print(all_pairs_shortest_paths)

    apsp = floyd_warshall(n, distances)
    delivery_graph = build_delivery_network(n, apsp, speeds, capacities)

    result_map = {}
    for u, v in queries:
        if u not in result_map:
            result_map[u] = dijkstra(n, delivery_graph, u)

    answers = []
    for u, v in queries:
        time = result_map[u][v]
        answers.append("INF" if math.isinf(time) else f"{time:.3f}")

    print(" ".join(answers))

