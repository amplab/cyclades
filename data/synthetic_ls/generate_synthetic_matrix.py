from __future__ import print_function
import sys
import random

output_file = sys.argv[1]
n_datapoint = int(sys.argv[2])
n_model_coordinates = int(sys.argv[3])
sparsity_percentage = float(sys.argv[4])
f_out = open(output_file, "w")

print("%d" % n_model_coordinates, file=f_out)
sample_set = range(0, n_model_coordinates)
for i in range(n_datapoint):
    n_coordinates = int(random.uniform(0, 1) * n_model_coordinates * sparsity_percentage)
    nnz_values = [random.uniform(1, 1000) for ii in range(n_coordinates)]
    indices = random.sample(sample_set, n_coordinates)
    rest_str = " ".join([str(indices[ii]) + " " + str(nnz_values[ii]) for ii in range(n_coordinates)])
    line_str = "%d %s" % (i, rest_str)
    print(line_str, file=f_out)
f_out.close()
