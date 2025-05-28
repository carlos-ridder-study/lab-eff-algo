import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class FindBridges {
    private static List<List<Integer>> graph;
    private static int[] discoveryTime, lowLink, parent;
    private static boolean[] visited;
    private static int time;
    private static int bridgeCount;

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        String[] firstLine = br.readLine().split(" ");
        int n = Integer.parseInt(firstLine[0]);
        int m = Integer.parseInt(firstLine[1]);

        // Initialize Graph
        graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }

        for (int i = 0; i < m; i++) {
            String[] edge = br.readLine().split(" ");
            int u = Integer.parseInt(edge[0]);
            int v = Integer.parseInt(edge[1]);
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        // Initialize Arrays
        discoveryTime = new int[n];
        Arrays.fill(discoveryTime, -1);
        lowLink = new int[n];
        Arrays.fill(lowLink, -1);
        visited = new boolean[n];
        Arrays.fill(visited, false);
        parent = new int[n];
        Arrays.fill(parent, -1);
        time = 0;
        bridgeCount = 0;

        // Find bridges with DFS
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(i);
            }
        }
        System.out.println(bridgeCount);
    }

    private static void dfs(int node) {
        visited[node] = true;
        discoveryTime[node] = lowLink[node] = time++;

        for (int neighbor : graph.get(node)) {
            if (!visited[neighbor]) {
                parent[neighbor] = node;
                dfs(neighbor);

                lowLink[node] = Math.min(lowLink[node], lowLink[neighbor]);

                // Check if node is edge is a bridge
                if (lowLink[neighbor] > discoveryTime[node]) {
                    bridgeCount++;
                }
            } else if (neighbor != parent[node]) {
                lowLink[node] = Math.min(lowLink[node], lowLink[neighbor]);
            }
        }
    }
}
