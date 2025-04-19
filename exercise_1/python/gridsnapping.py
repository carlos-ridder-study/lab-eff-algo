# Grid Snapping  –  Efficient‑Algorithms Lab, Problem Set 1
import sys
from decimal import Decimal, ROUND_HALF_EVEN 

def dec_to_int(d: str) -> int:
    """return the value in integer; '0.75' → 75, '-2.40' → -240."""
    return int((Decimal(d) * 100))

def main() -> None:
    allread   = sys.stdin.read().strip().split()
    number_of_points    = int(allread[0])
    grid_width    = dec_to_int(allread[1])
    assert grid_width > 0, "Grid width must be positive."

    pts  = allread[2:]              # 2 · n remaining
    out_lines = []

    for i in range(0, len(pts), 2):
        x = dec_to_int(pts[i])
        y = dec_to_int(pts[i+1])

        # modulo goes down, even for negatives
        snap_x = (x // grid_width) * grid_width
        snap_y = (y // grid_width) * grid_width

        ## poroject back and make correct precision
        out_lines.append(f"{snap_x/100:.2f} {snap_y/100:.2f}")

    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    main()
