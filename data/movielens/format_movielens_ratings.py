from __future__ import print_function
import sys

ratings_file = sys.argv[1]
output_file = sys.argv[2]

f_ratings = open(ratings_file, "r")
o_file = open(output_file, "w")
datapoints, user_map, movie_map = [], {}, {}
n_users, n_movies = 1, 1
for line in f_ratings:
    vals = [float(x) for x in line.split("::")]
    keep_vals = [int(vals[0]), int(vals[1]), float(vals[2])]
    #datapoints.append(keep_vals[0:3])
    if keep_vals[0] not in user_map:
        user_map[keep_vals[0]] = n_users
        n_users += 1
    if keep_vals[1] not in movie_map:
        movie_map[keep_vals[1]] = n_movies
        n_movies += 1
    datapoints.append([user_map[keep_vals[0]], movie_map[keep_vals[1]], keep_vals[2]])

print("%d %d" % (len(user_map)+1, len(movie_map)+1), file=o_file)
print("%d %d" % (len(user_map)+1, len(movie_map)+1))
for datapoint in datapoints:
    print(" ".join([str(x) for x in datapoint]), file=o_file)
