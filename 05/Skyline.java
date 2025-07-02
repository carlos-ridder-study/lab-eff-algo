import java.util.*;

import static java.lang.Math.abs;

public class Skyline {

    static Map<Integer, List<Integer>> skyline = new HashMap<>();

    // Manhattan distance between two points given the nxn grid size.
    static int distance(int p1, int p2, int n) {
        return abs(p1 % n - p2 % n) + abs(p1 / n - p2 / n);
    }

    // Add position integer according to value.
    // return null if no other element has this value, return the minimum distance between all those elements otherwise.
    static Integer add(int position, int value, int n){
        if (skyline.get(value) == null) {
            List<Integer> list = new ArrayList<>();
            list.add(position);
            skyline.put(value, list);
            return null;
        } else {
            int distance = Integer.MAX_VALUE;
            skyline.get(value).add(position);
            List<Integer> list = skyline.get(value);
            int lastPosition = list.size() - 1;
            int lastComparison = 0;
            for (int i = lastPosition - 1; i >= 0; i--) {
                int comp = list.get(i);
                if (comp < lastComparison) { //smallest possible position: ((dist-1) * n) behind last found distance;
                    break;
                }
                if (distance(comp, position, n) < distance) {
                    distance = distance(comp, position, n);
                    lastComparison = comp - ((distance - 1) * n);
                }
            }
            return distance;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        int nextValue;
        int distance = Integer.MAX_VALUE;
        Integer nextDistance;
        int position = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                nextValue = sc.nextInt();
                nextDistance = add(position, nextValue, n);
                if (nextDistance != null && nextDistance < distance) {
                    distance = nextDistance;
                }
                position++;
                if (distance == 1) {
                    System.out.println(distance);
                    return;
                }
            }
        }
        System.out.println(distance);
    }
}
