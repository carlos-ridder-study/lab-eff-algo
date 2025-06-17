import java.util.*;
import java.util.stream.IntStream;

public class Tyrion {

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int n = Integer.parseInt(input.nextLine());

        Map<String, Integer> houseIndex = new HashMap<>();
        List<BitSet> A = new ArrayList<>();
        List<Integer> b = new ArrayList<>();
        int varCount = 0; //keep track of house index

        for (int i = 0; i < n; i++) {
            String[] parts = input.nextLine().trim().split("\\s+");
            int parity = parts[parts.length - 1].equals("odd") ? 1 : 0;

            BitSet row = new BitSet();
            for (int j = 0; j < parts.length - 1; j++) {
                String house = parts[j];
                if (houseIndex.putIfAbsent(house, varCount) == null) varCount++;
                row.set(houseIndex.get(house));
            }
            A.add(row);
            b.add(parity);
        }
        int m = varCount;
        int rows = A.size();
        BitSet[] matrix = A.toArray(new BitSet[0]);
        int[] rhs = b.stream().mapToInt(i -> i).toArray();

        int row = 0;
        for (int col = 0; col < m; col++) {
            int pivot = -1;
            for (int i = row; i < rows; i++) {
                if (matrix[i].get(col)) {
                    pivot = i;
                    break;
                }
            }
            if (pivot == -1) {
                continue;
            }

            // swap row and pivot
            BitSet tmp = matrix[row];
            matrix[row] = matrix[pivot];
            matrix[pivot] = tmp;
            int temp = rhs[row];
            rhs[row] = rhs[pivot];
            rhs[pivot] = temp;

            for (int i = 0; i < rows; i++) {
                if (i != row && matrix[i].get(col)) {
                    matrix[i].xor(matrix[row]);
                    rhs[i] ^= rhs[row];
                }
            }
            row++;
        }

        int solution = IntStream.of(rhs).sum();
        System.out.println(IntStream.of(solution).sum());
    }
}
