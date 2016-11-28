wget -qO- http://www.mattmahoney.net/dc/enwik8.zip | jar xvf /dev/stdin
perl wikifil.pl enwik8 > cleaned_corpus
python create_graph_from_corpus.py cleaned_corpus
