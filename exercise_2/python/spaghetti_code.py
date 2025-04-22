import sys

# read in lines from standard input
line = sys.stdin.readline().strip().split(" ")
a = int(line[0])
b  = int(line[1])

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
if week < threshold:
    time = a*week
elif week >= threshold:
    time = a*threshold + b
print(time)