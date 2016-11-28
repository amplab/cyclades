wget -qO- http://files.grouplens.org/datasets/movielens/ml-1m.zip | jar xvf /dev/stdin && python format_movielens_ratings.py ml-1m/ratings.dat ml-1m/movielens_1m.data
wget -qO- http://files.grouplens.org/datasets/movielens/ml-10m.zip | jar xvf /dev/stdin && python format_movielens_ratings.py ml-10M100K/ratings.dat ml-10M100K/movielens_1m.data
