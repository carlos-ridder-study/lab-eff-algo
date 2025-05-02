import java.io.*;
import java.util.*;
import java.math.BigDecimal;

public class KMeans1D {
    public static void main(String[] args) throws IOException {
        
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        //first line: numberofCenters numberOfQueries accaptableRange
        String[] firstLine = br.readLine().trim().split("\\s+");
        int numberOfCenters = Integer.parseInt(firstLine[0]);
        int numberOfQueries = Integer.parseInt(firstLine[1]);
        BigDecimal R = new BigDecimal(firstLine[2]);

        // sort centers
        BigDecimal[] centers = new BigDecimal[numberOfCenters];
        for (int i = 0; i < numberOfCenters; i++) {
            centers[i] = new BigDecimal(br.readLine().trim());
        }
        Arrays.sort(centers);

        // process queries with binary search
        for (int i = 0; i < numberOfQueries; i++) {
            BigDecimal query = new BigDecimal(br.readLine().trim());
            BigDecimal closest = findClosestInRange(centers, query, R);
            if (closest == null) {
                bw.write("none in range\n");
            } else {
                bw.write(String.format("%.2f\n", closest));
            }
        }

        bw.flush();
    }

    // binary search + check neighbors to find closest center within r
    private static BigDecimal findClosestInRange(BigDecimal[] centers, BigDecimal query, BigDecimal R) {
        int idx = Arrays.binarySearch(centers, query);

        if (idx >= 0) {
            return centers[idx]; // exact match
        }

        // if no match, idx is the position it would fit if inserted
        // so convert back to correct index to check the elements
        idx = -idx - 1;

        BigDecimal best = null; // if non found then give back signal value of null
        BigDecimal minDist = null;

        if (idx < centers.length) { // check bigger center first
            BigDecimal dist = centers[idx].subtract(query).abs();
            if (dist.compareTo(R) <= 0) {
                best = centers[idx];
                minDist = dist;
            }
        }

        if (idx > 0) { // now center before insertion point, need to do this after so we can pick this one in case of tie
            BigDecimal dist = centers[idx - 1].subtract(query).abs();
            if (dist.compareTo(R) <= 0) {
                if (minDist == null || dist.compareTo(minDist) < 0 ||
                    (dist.compareTo(minDist) == 0 && centers[idx - 1].compareTo(best) < 0)) { //tie break condition
                    best = centers[idx - 1];
                    minDist = dist;
                }
            }
        }

        return best;
    }
}
