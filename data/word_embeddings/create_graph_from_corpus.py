from __future__ import print_function
import sys
from random import shuffle

DISTANCE = 10

with open(sys.argv[1], 'r') as corpus:
    text = corpus.read()
    text = text[:1000000]

    words_list = list(set(text.split()))
    word_to_id = {}

    # Write word id mappings
    for index, word in enumerate(list(set(words_list))):
        word_to_id[word] = index

    # Construct graph
    g = {}
    words = text.strip().split(" ")
    lines = [words[i:i+DISTANCE] for i in range(len(words))]
    for line in lines:
        if len(line) < DISTANCE:
            continue
        #print("LINE: ", line)
        first_word = line[0]
        for other_word in line:
            if other_word == first_word:
                continue
            a, b = tuple(sorted([first_word, other_word]))
            if (a,b) not in g:
                g[(a, b)] = 0
            g[(a, b)] += 1

    # Output graph to file
    f = open("w2v_graph", "w")
    n_nodes = len(set(list(words_list)))
    n_edges = len(g.items());
    print("%d" % n_nodes, file=f)
    for word_pair, occ in g.items():
        print("%d %d %d" % (word_to_id[word_pair[0]], word_to_id[word_pair[1]], occ), file=f)

    # Print stats
    print("N_NODES=%d" % n_nodes)
    print("N_EDGES=%d" % n_edges)
    f.close()
