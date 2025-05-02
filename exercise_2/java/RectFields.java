import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;

public class RectFields {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        int rows = Integer.parseInt(reader.readLine().trim());

        //row wise strings into chars arrays
        String[] grid = new String[rows];
        for (int i = 0; i < rows; i++) {
            grid[i] = reader.readLine().trim();
        }

        int columns = grid[0].length();  
        int[][] binary = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            char[] charRow = grid[i].toCharArray();
            for (int j = 0; j < columns; j++) {
                binary[i][j] = charRow[j] - '0'; //ascii subtract to make it either 1 or 0
            }
        }

        //dont know a smarter way than to quadratically iterate row pairs
        long count = 0;
        for (int row1 = 0; row1 < rows; row1++) {
            for (int row2 = row1 + 1; row2 < rows; row2++) {
                int verticals = 0;
                for (int col = 0; col < columns; col++) {
                    if (binary[row1][col] == 1 && binary[row2][col] == 1) {
                        verticals++;
                    }
                }
                // with all column aligned pairs, we an calc the combinations with verticals choose 2 without permutation
                count += (long) verticals * (verticals - 1) / 2;
            }
        }

        System.out.println(count);
    }
}
