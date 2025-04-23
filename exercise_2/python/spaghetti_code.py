import sys

# read in lines from standard input
a = int(sys.stdin.readline().strip())
b  = int(sys.stdin.readline().strip())

# algorithm goes here
threshold = b // a - 1

line = sys.stdin.readline().strip()
week = 0
while line != "stop":
    if week < threshold:
        print("FIX")
    elif week == threshold:
        print("REFACTOR")
    elif week > threshold:
        print("NOTHING")
    sys.stdout.flush()
    line = sys.stdin.readline().strip()
    week += 1

# output time spent
time = 0
if week <= threshold:
    time = a*week
else:
    time = a*threshold + b
print(time)