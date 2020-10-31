#! /usr/bin/python3

import os
from author import Author, Tokenizer
from train import split_data_into_train_test
from textspan import TextSpan
import textutils
from embedding import EmbeddingModel
from smarttimers import SmartTimer


######################
# User Configuration #
######################
infile = '../data/Doyle_10.txt'
outdir = 'tmp'
part_size = 3500


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
a = Author(infile)
t.toc()
print('Corpus characters:', len(a.corpus))

t.tic('Preprocessing: Tokenizer')
a.preprocess(Tokenizer())
t.toc()
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))
print('Corpus vocabulary:', len(a.parsed.vocabulary))

t.tic('Document partitioning')
a.partition_into_docs(part_size)
t.toc()
print('Documents:', len(a.docs))
print('Document tokens:', a.docs[0].size)

t.tic('Author embedding')
m = EmbeddingModel()
m.train(a.sentences)
t.toc()
print('Embedding vocabulary:', len(m.vocabulary))
print('Embedding matrix:', m.vectors.shape)

print('Walltime:', t.walltime)
print(t)
