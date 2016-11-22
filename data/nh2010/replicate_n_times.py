from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Expect matrix market format. If A is the matrix, replicates like:
# [A, A ... ; A A .. ], so for n = 2, writes [A, A; A, A]

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def visualize(sparse_mat, n_rows_cols):
    # make symm
    reflected = [(x[1], x[0], x[2]) for x in sparse_mat]
    sparse_mat += reflected
    vals = [x[2] for x in sparse_mat]
    rows = [x[0] for x in sparse_mat]
    cols = [x[1] for x in sparse_mat]
    all_rows_cols = rows+cols
    m = coo_matrix((vals, (rows, cols)), shape=(n_rows_cols,n_rows_cols))
    ax = plot_coo_matrix(m)
    ax.figure.show()
    plt.show()

input_file = sys.argv[1]
output_file = sys.argv[2]
n = int(sys.argv[3])
f_in = open(input_file, "r")
f_out = open(output_file, "w")

all_values = []
first_line = True
n_rows, n_cols, n_entries = 0, 0, 0
for line in f_in:
    if line[0] == "%":
        continue
    if first_line:
        first_line = False
        n_rows, n_cols, n_entries = [int(x) for x in line.strip().split()]
        continue
    weight = 1
    values = [float(x) for x in line.strip().split()]
    if len(values) == 3:
        c1, c2, weight = values
    else:
        c1, c2 = values
    all_values.append((c1-1,c2-1,weight))

if n_rows != n_cols:
    print("Must be square!")
    sys.exit(0)

#visualize(all_values, n_rows)

new_values = []
for value in all_values:
    dup_values = []
    for i in range(n):
        for j in range(n):
            dup_values.append((value[0] + i * n_rows, value[1] + j * n_rows, value[2]))
    new_values += dup_values

#visualize(new_values, n_rows * n)

print("%d %d %d" % (n_rows * n, n_cols * n, n_entries * n * n), file=f_out)
for value in new_values:
    print("%d %d %d" % (value[0]+1, value[1]+1, value[2]), file=f_out)
