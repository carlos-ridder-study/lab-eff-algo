import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class FastAndFuriousII {
    static class Edge { // Use class edge to make graph easier to deal with.
        int v; //edge leads to v
        int weight;

        Edge(int v, int weight) {
            this.v = v;
            this.weight = weight;
        }
    }

    private static List<List<Edge>> graph; // Use edge rather than Integers for easier handling later on.

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line = br.readLine();
        int n = Integer.parseInt(line.split(" ")[0]);
        int m = Integer.parseInt(line.split(" ")[1]);
        int d = Integer.parseInt(line.split(" ")[2]);

        // Initialize Graph
        graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }

        String[] road;
        for (int i = 0; i < m; i++) {
            road = br.readLine().split(" ");
            int a = Integer.parseInt(road[0]);
            int b = Integer.parseInt(road[1]);
            int w = Integer.parseInt(road[2]);
            graph.get(a).add(new Edge(b, w));
            graph.get(b).add(new Edge(a, w));
        }

        // The destinations we have to visit.
        String[] dest = br.readLine().split(" ");
        int[] destinations = new int[d];
        for (int i = 0; i < d; i++) {
            destinations[i] = Integer.parseInt(dest[i]);
        }

        // Construct a grid for the distances between each pair of destinations.
        int[][] distBetweenDestinations = new int[d][d];
        for (int i = 0; i < d; i++) {
            int[] distFromi = dijkstra(destinations[i], n);
            for (int j = 0; j < d; j++) {
                distBetweenDestinations[i][j] = distFromi[destinations[j]];
            }
        }

        // Use Dynamic Programming with bitmasking for an efficient solution to the TSP
        int[][] dp = new int[1 << d][d]; // 0001 -> 1000 for d:=3
        for (int[] row : dp) Arrays.fill(row, Integer.MAX_VALUE);
        dp[1][0] = 0; //start from city 0

        for (int mask = 1; mask < (1 << d); mask++) {
            for (int u = 0; u < d; u++) {
                if ((mask & (1 << u)) == 0 || dp[mask][u] == Integer.MAX_VALUE) continue;
                for (int v = 0; v < d; v++) {
                    if ((mask & (1 << v)) == 0) { //is v-th bit set? //reversed logic better?
                        int nextMask = mask | (1 << v); //set v-th bit
                        dp[nextMask][v] = Math.min(dp[nextMask][v], dp[mask][u] + distBetweenDestinations[u][v]);
                    }
                }
            }
        }
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < d; i++) {
            if (dp[(1 << d) - 1][i] != Integer.MAX_VALUE && distBetweenDestinations[i][0] != Integer.MAX_VALUE) {
                result = Math.min(result, dp[(1 << d) - 1][i] + distBetweenDestinations[i][0]); //pick shortest route
            }
        }
        System.out.println(result);
    }


    // Use Dijkstra's Algorithm to compute the shortest route from start node to all other nodes.
    static int[] dijkstra(int start, int n) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        dist[start] = 0;
        pq.add(new int[]{start, 0});

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int u = curr[0];
            int d = curr[1];
            if (d >= dist[u]) {
                for (Edge e : graph.get(u)) { // Using class Edge here makes things easier for iterating through edges.
                    if (dist[e.v] > dist[u] + e.weight) {
                        dist[e.v] = dist[u] + e.weight;
                        pq.add(new int[]{e.v, dist[e.v]});
                    }
                }
            }
        }
        return dist;
    }
}