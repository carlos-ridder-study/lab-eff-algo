import sys
import time
import string


# for debugging with input file
import os
input_file = os.environ.get('INPUT_FILE')
if input_file:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the input file
    input_path = os.path.join(script_dir, input_file)
    sys.stdin = open(input_path, 'r')
     
def de_bruijn(k: int, n: int) -> str:
    """
    Generate a De Bruijn sequence for alphabet range(k) and subsequences of length n.
    Returns a string of length k**n containing every substring of length n exactly once (cyclically).
    """
    
    # special cases for which integer arithmetic does not work
    if n == 0 or k == 0:
        return []
    if n == 1:
        return [i for i in range(k)]
    if k == 1:
        return [0] * n

    # Number of distinct (n-1)-length nodes
    base = k ** (n - 1)

    # Build adjacency lists: for each node v in (0 .. base-1),
    # store the list of outgoing digits [0..k-1] 
    adj = [list(range(k))[::-1] for _ in range(base)]

    # Hierholzer’s stack and circuit
    stack = [0]       # start at node 0 (which represents (n-1)*"0")
    circuit = []      # store nodes on euler circuit in reverse order

    """
    Depth-first edge traversal of de-bruijn graph with nodes corresponding to substrings of length n-1
    and edges to substrings of length n. This constructs an euler circuit of the graph. 
    
    Correctness:
    1. each edge is traversed exactly ones and its endpoint is pushed to the stack, because the graph is connected.
       
    2. all nodes have in- and out- degree k as they have exactly one neighbor for any digit in range(k) that can be 
       appended to the right after shifting the substring to the left and discarding the leading digit.
       
       this means that the algorithm does not get stuck but always constructs circles on the stack, because any interior node
       on the path always has the same in- and out-degree, so if it has been reached by an unused edge it can also
       be left by an unused edge. 
       in contrast the starting node always has || has out-degree = in-degree - 1 || after it was left on the path, so it will be 
       the first without any unseen outgoing edges left and added to the circuit.
       then a new path gets constructed from the node on top of the stack, which is the node through which the 
       first node was reached just before. now the same reasoning applies for this node as its out-degree is always
       one less than its in-degree after it was left on the path and it gets appended to the circuit next.
       inductively this proves that the circuit always contains a path in reverse order.
       
    This concludes the proof, because the algorithm runs until the stack is empty and the endpoint of each edge will
    eventually be pushed on the stack and then popped and appended to the circuit, which therefore contains all
    edges of the graph forming a path in reverse order.
    """
    while stack:
        v = stack[-1]
        if adj[v]:
            # take next outgoing edge (digit c)
            c = adj[v].pop()
            # compute destination node: shift left in base-k, add c, mod base to discard highest digit
            u = (v * k + c) % base
            # add node to the stack representing the edge v -> u
            stack.append(u)
        else:
            # no more edges: backtrack
            circuit.append(stack.pop())

    circuit.reverse()

    sequence = [node % k for node in circuit[:-1]]
    return sequence


def map_sequence(seq: list[int], symbols: list[str]) -> str:
    """
    Map each character in `seq` (which must be in range(len(symbols)) )
    to the symbol at that index in `symbols`.
    """
    k = len(symbols)
    out = [symbols[i] for i in seq]
    return "".join(out)


def validate_de_bruijn_sequence(sequence: str, alphabet: list[str], n: int) -> bool:
    """
    Validate if every substring of length `n` from the given `alphabet` is present in the `sequence`.

    Args:
        sequence (str): The sequence to validate.
        alphabet (list[str]): The list of symbols in the alphabet.
        n (int): The length of substrings to check.

    Returns:
        bool: True if the sequence is valid, False otherwise.
    """
    from itertools import product

    # Generate all possible substrings of length n from the alphabet
    expected = {"".join(p) for p in product(alphabet, repeat=n)}
    
    # Use a sliding window over the sequence to get all substrings of length n.
    observed = set(sequence[i:i+n] for i in range(len(sequence) - n + 1))
    
    missing = expected - observed
    if missing:
        for substring in missing:
            print(f"Missing substring: {substring}")
        return False
    return True

# read in lines from standard input
line = sys.stdin.readline().strip().split(" ")
n = int(line[0])
k = int(line[1])

# algorithm goes here
alphabet = list(string.ascii_lowercase[:k])
flowers = ["🌹", "🌼", "🌻", "🌸", "🌷", "🌺", "🪻", "💮", "🍀", "✿"]

time_start = time.time()
cyclic_sequence = de_bruijn(k, n)
answer = cyclic_sequence
if k > 1:
    answer += cyclic_sequence[:n-1]
# flowers are nicer but servers does not accept them
# answer = map_sequence(answer, alphabet[:k])
answer = map_sequence(answer, flowers[:k])
print(answer)

"""
# debugging
print("optimum k**n + n-1 = ", k**n+n-1, ", answer has length",len(answer))
time_end = time.time()

print("time:", time_end - time_start)

# Example usage to validate the program's output
if __name__ == "__main__":
    is_valid = validate_de_bruijn_sequence(answer, flowers[:k], n)
    if is_valid:
        print("The answer is correct.")
"""