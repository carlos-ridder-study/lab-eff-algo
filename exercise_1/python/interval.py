import sys

line = sys.stdin.readline().strip()
value = line.split(" ")
a = int(value[0])
b = int(value[1])
k = int(value[2])

if b >= k * a:
    deleted = b // k - a + 1
else:
    deleted = 0
    
print(deleted)