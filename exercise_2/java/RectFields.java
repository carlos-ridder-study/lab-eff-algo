import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.BitSet;

public class RectFields {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        int rows = Integer.parseInt(reader.readLine().trim());

        // row wise strings into chars arrays
        String[] grid = new String[rows];
        for (int i = 0; i < rows; i++) {
            grid[i] = reader.readLine().trim();
        }

        int columns = grid[0].length();

        //bitSet instead of int for bitwise operations
        BitSet[] binary = new BitSet[rows];
        for (int i = 0; i < rows; i++) {
            binary[i] = new BitSet(columns);
            char[] charRow = grid[i].toCharArray();
            for (int j = 0; j < columns; j++) {
                if (charRow[j] == '1') {
                    binary[i].set(j);
                }
            }
        }

        // don't know a smarter way than to quadratically iterate row pairs
        long count = 0;
        for (int row1 = 0; row1 < rows; row1++) {
            for (int row2 = row1 + 1; row2 < rows; row2++) {
                
                BitSet intersection = (BitSet) binary[row1].clone();
                intersection.and(binary[row2]);
                int verticals = intersection.cardinality();

                // with all column aligned pairs, we can calc the combinations with verticals choose 2 without permutation
                count += (long) verticals * (verticals - 1) / 2;
            }
        }

        System.out.println(count);
    }
}