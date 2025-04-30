import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.BitSet;
import java.util.Random;


public class RectangularFields {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int r = Integer.parseInt(br.readLine());
        // Test Benchmark
        if (r < 0) {
            r = 2000;
            BitSet[] orchid = new BitSet[r];
            Random rand = new Random();
            for (int i = 0; i < r; i++) {
                orchid[i] = new BitSet(r);
                for (int j = 0; j < r; j++) {
                    if (rand.nextBoolean()) {
                        orchid[i].set(j);
                    }
                }
            }
            long start = System.currentTimeMillis();
            System.out.println(countRectangles(orchid, r));
            long end = System.currentTimeMillis();
            System.out.print(end - start + " ms");
        }
        else {
        BitSet[] orchid = new BitSet[r];
        String line;
        for (int i = 0; i < r; i++) {
            line = br.readLine();
            orchid[i] = new BitSet(r);
            for (int j = 0; j < r; j++) {
                if (line.charAt(j) == '1') {
                    orchid[i].set(j);
                }
            }
        }
        System.out.print(countRectangles(orchid, r));
        }
    }

    static int countRectangles(BitSet[] orchid, int r) {
        int count = 0, bits = 0, rowPairs = 0;
        for (int i = 0; i < r; i++) { // col1
            for (int j = i + 1; j < r; j++) { // col2
                BitSet commonRows = (BitSet) orchid[i].clone();
                commonRows.and(orchid[j]);
                bits = commonRows.cardinality();
                rowPairs = bits * (bits - 1);
                count += rowPairs / 2;
            }
        }
        return count;
    }
}
