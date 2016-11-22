from __future__ import print_function
import sys

# Given a matrix in Matrix Market format, converts it into the following format:
# Line 1: # of rows
# Line 2..n: Row# col1_# col1_weight col2_# col2_weight ... coln_# coln_weight

input_file = sys.argv[1]
output_file = sys.argv[2]
f = open(input_file)
f_out = open(output_file, "w")
node_id, node_count = {}, 0
m = {}
for line in f:
    if line[0] == "%":
        continue
    weight = 1
    values = [float(x) for x in line.strip().split()]
    if len(values) == 3:
        c1, c2, weight = values
    else:
        c1, c2 = values
    if c1 not in node_id:
        node_id[c1] = node_count
        node_count += 1
    if c2 not in node_id:
        node_id[c2] = node_count
        node_count += 1

    if node_id[c1] not in m:
        m[node_id[c1]] = set()
    if node_id[c2] not in m:
        m[node_id[c2]] = set()
    m[node_id[c1]].add((node_id[c2], weight))
    m[node_id[c2]].add((node_id[c1], weight))

print("%d" % node_count, file=f_out)
for i in range(node_count):
    index_weight_pairs = [str(x[0]) + " " + str(x[1]) for x in m[i]]
    line_str = str(i) + " " + " ".join(index_weight_pairs)
    line_str = line_str.strip()
    print(line_str, file=f_out)

f.close()
