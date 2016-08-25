from __future__ import print_function
import sys
from math import sqrt

N = int(sys.argv[1])
out_fname = sys.argv[2]
f_out = open(out_fname, "w")

if int(sqrt(N)) ** 2 != N:
    print("Error: N must be a square.")
    sys.exit(0)

n = int(sqrt(N))
m = {}
for i in range(n):
    for j in range(n):
        index = i * n + j
        if index not in m:
            m[index] = set()
        for ii in [-1, 0, 1]:
            for jj in [-1, 0, 1]:
                # Only horizontal and vertical connections.
                if abs(ii) + abs(jj) > 1:
                    continue
                new_i, new_j = i+ii, j+jj
                if new_i < 0 or new_j < 0 or new_i >= n or new_j >= n:
                    continue
                neighbor_index = new_i * n + new_j
                if neighbor_index not in m:
                    m[neighbor_index] = set()
                if neighbor_index != index:
                    m[index].add(neighbor_index)
                    m[neighbor_index].add(index)

print("%d 1" % N, file=f_out)
for i in range(N):
    neighbor_string = " ".join([str(x) for x in list(m[i])])
    print("%d %s" % (i, neighbor_string), file=f_out)
f_out.close()
