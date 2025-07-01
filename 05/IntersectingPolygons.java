import java.util.*;

public class IntersectingPolygons {

    static class Point{
        double x, y;
        Point(double x, double y){
            this.x = x;
            this.y = y;
        }
    }


    static boolean isInPolygon(List<Point> polygon, double x, double y){
        int n = polygon.size();
        boolean inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            Point p1 = polygon.get(i);
            Point p2 = polygon.get(j);
            boolean intersect = ((p1.y > y) != (p2.y > y)) &&
                    (x < (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y + 1e-9) + p1.x);
            if (intersect) {
                inside = !inside;
            }
        }
        return inside;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        List<List<Point>> polygons = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int k = sc.nextInt();
            List<Point> polygon = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                polygon.add(new Point(sc.nextInt(), sc.nextInt()));
            }
            polygons.add(polygon);
        }

        double area = 0;

        // search the grid with step size 0.01
        for (double x = 0.0; x <= 10.00; x += 0.01) {
            for(double y = 0.0; y <= 10.00; y += 0.01) {
                boolean insideAll = true;
                for(List<Point> polygon : polygons) {
                    if (!isInPolygon(polygon, x, y)) {
                        insideAll = false;
                        break;
                    }
                }
                if (insideAll) {
                    area += 0.0001; //0.01^2
                }
            }
        }
        System.out.println(area);

        boolean test = false;
        if (test) {
            List<Point> points = new ArrayList<>();
            points.add(new Point(4, 2)); //boundary             false false
            points.add(new Point(4, 1)); //boundary of polygon1 false true
            points.add(new Point(3.8, 1.2)); //inside both      true true
            points.add(new Point(3, 1.001)); //inside polygon1  true false
            points.add(new Point(4, 0.999)); //inside polygon2  false true
            points.add(new Point(4.001, 2)); //outside both     false false

            //long time = System.currentTimeMillis();
            //long time = System.nanoTime();
            for (Point point : points) {
                for (List<Point> polygon : polygons) {
                    System.out.println(isInPolygon(polygon, point.x, point.y));
                }
            }
            //System.out.println(System.nanoTime() - time);
        }
    }
}
