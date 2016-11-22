from __future__ import print_function
from scipy import sparse
import sys
import random
from scipy.linalg import block_diag

output_file = sys.argv[1]
n_blocks = int(sys.argv[2])
block_size = int(sys.argv[3])
f_out = open(output_file, "w")
blocks = []

for i in range(n_blocks):
    cur_block = []
    for j in range(block_size):
        cur_block.append([])
        for k in range(block_size):
            cur_block[-1].append(random.uniform(1, 1000))
    blocks.append(cur_block)

result = block_diag(*blocks)
sparse_result = sparse.coo_matrix(result)
print(result)

print("%d" % len(result[0]), file=f_out)
m = {}
for i,j,v in zip(sparse_result.row, sparse_result.col, sparse_result.data):
    if i not in m:
        m[i] = {}
    m[i][j] = v

for i in range(len(result)):
    row = m[i]
    rest_str = " ".join([str(ind) + " " + str(val) for ind,val in row.items()])
    line_str = "%d %s" % (i, rest_str)
    print(line_str, file=f_out)
f_out.close()
