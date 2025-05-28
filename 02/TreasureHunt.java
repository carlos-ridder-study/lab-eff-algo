import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class TreasureHunt {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine());
        int m = Integer.parseInt(br.readLine());
        int[][] matrix = new int[n][m];
        String[] line;
        for (int i = 0; i < n; i++) {
            line = br.readLine().split(" ");
            for (int j = 0; j < m; j++) {
                matrix[i][j] = Integer.parseInt(line[j]);
            }
        }

        //Compute maximum path sum
        int[][] sum = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                sum[i][j] = Math.max(sum[i - 1][j], sum[i][j - 1])
                        + matrix[i - 1][j - 1];
            }
        }
        System.out.println(sum[n][m]);
    }
}
