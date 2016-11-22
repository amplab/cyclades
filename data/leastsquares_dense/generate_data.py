from __future__ import print_function
import sys
import random
import numpy as np

n_rows, n_parameters = tuple([int(x) for x in [sys.argv[1], sys.argv[2]]])
outfilename = sys.argv[3]

f_out = open(outfilename, "w")
print(n_parameters, file=f_out)
for i in range(n_rows):
    random_values = np.random.normal(0, 1, n_parameters)
    random_label = random.randint(-1, 1)
    result = []
    result.append(random_label)
    for index,value in enumerate(random_values):
        result.append(index)
        result.append(value)
    print(" ".join([str(x) for x in result]), file=f_out)
