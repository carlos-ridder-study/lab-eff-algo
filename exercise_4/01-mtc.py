import sys
from collections import defaultdict, deque

# for debugging with input file
import os
input_file = os.environ.get('INPUT_FILE')
if input_file:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the input file
    input_path = os.path.join(script_dir, input_file)
    sys.stdin = open(input_path, 'r')


class Kosaraju_SCC:
    """Class to find strongly connected components using Kosaraju's algorithm.
    This class uses a directed graph represented as an adjacency list.
    """
    def __init__(self, n, graph=None):
        self.n = n
        self.graph = graph
        self.counter = 0
        self.index = [-1] * n
        self.lowlink = [-1] * n
        self.on_stack = [False] * n
        self.stack = []
        
    def _dfs(self, v, scc_nodes, scc_id, scc_size):
        stack = []
        stack.append((v, 0))
        while stack:
            node, state = stack.pop()
            if state == 0:
                if self.index[node] == -1:
                    self.index[node] = self.lowlink[node] = self.counter
                    self.counter += 1
                    self.stack.append(node)
                    self.on_stack[node] = True
                    stack.append((node, 1))
                    for w in self.graph[node]:
                        if self.index[w] == -1:
                            stack.append((w, 0))
                        elif self.on_stack[w]:
                            self.lowlink[node] = min(self.lowlink[node], self.index[w])
            else:
                for w in self.graph[node]:
                    if self.index[w] != -1 and self.on_stack[w]:
                        self.lowlink[node] = min(self.lowlink[node], self.lowlink[w])
                if self.lowlink[node] == self.index[node]:
                    scc_size[node] = 0
                    while True:
                        w = self.stack.pop()
                        self.on_stack[w] = False
                        scc_size[node] += 1
                        scc_id[w] = node
                        if w == node:
                            break
                    scc_nodes.append(node)
        
    def computeSCCgraph(self):
        """Computes the strongly connected components of the graph in reverse topological order."""
        scc_id = [-1] * self.n
        scc_size = defaultdict(int)
        scc_nodes = []
        scc_edges = defaultdict(set)
        self.counter = 0
        self.index = [-1] * n
        self.lowlink = [-1] * n
        self.on_stack = [False] * n
        self.stack = []
        for v in range(self.n):
            if self.index[v] == -1:
                self._dfs(v, scc_nodes, scc_id, scc_size)
                
        # add edges between strongly connected components
        for u in range(self.n):
            for v in self.graph[u]:
                if scc_id[u] != scc_id[v]:
                    if scc_id[v] not in scc_edges[scc_id[u]]:
                        scc_edges[scc_id[u]].add(scc_id[v])
        return (scc_nodes, scc_edges), scc_size

# read in lines from standard input
line = sys.stdin.readline().strip().split(" ")
n = int(line[0])  # number of nodes
m  = int(line[1])  # number of edges
graph = [[] for _ in range(n)]
for _ in range(m):
    line = sys.stdin.readline().strip()
    u, v = map(int, line.split())
    graph[u].append(v)


"""
import random
n = 50000
graph = [[] for _ in range(n)]
for u in range(n):
    num_edges = random.randint(0, min(10, n-1))
    # Choose unique targets, excluding self
    targets = set()
    while len(targets) < num_edges:
        v = random.randint(0, n - 1)
        if v != u:
            targets.add(v)
    graph[u].extend(targets)
"""


# algorithm goes here
scc = Kosaraju_SCC(n, graph)
scc_graph, scc_size = scc.computeSCCgraph()

# compute maximum chain in scc_graph with scc_sizes as node weights
dp = {u: scc_size[u] for u in scc_graph[0]}
for u in reversed(scc_graph[0]):
    for v in scc_graph[1][u]:
        dp[v] = max(dp[v], dp[u] + scc_size[v])

# print the result        
result = max(dp.values())
print(result)