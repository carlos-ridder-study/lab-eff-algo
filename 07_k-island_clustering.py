import sys


def max_spacing_k_clustering(n, k, edges):
    # Sort edges (u,v,w) by weight
    edges.sort(key=lambda x: x[2])

    parent = list(range(n))

    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # Test if adding the edge to the current forest would create a cycle
    def cyclical(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:  # if u, v have the same parent, a cycle results
            parent[rv] = ru
            return True
        return False

    # Compute MST of the graph with Kruskal's algorithm
    mst_edges = []
    for u, v, w in edges:
        if cyclical(u, v):
            mst_edges.append((u, v, w))  # append edge bc it does not create a cycle

    # Remove (k-1) largest edges from MST -> k clusters remaining
    mst_edges.sort(key=lambda x: -x[2])
    answer = mst_edges[k - 2][2]  # (k-1)th largest edge weight
    return answer


# Read in values
n = int(sys.stdin.readline().strip())
k = int(sys.stdin.readline().strip())
m = int(sys.stdin.readline().strip())

# edge: vertice u, vertice v, weight w
edges = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(m)]

print(max_spacing_k_clustering(n, k, edges))
