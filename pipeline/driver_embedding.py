#! /usr/bin/python3

import os
from author import Author, Tokenizer
from train import split_data_into_train_test
from textspan import TextSpan
import textutils
from embedding import EmbeddingModel


######################
# User Configuration #
######################
infile = '../data/Doyle_10.txt'
outdir = 'tmp'
part_size = 3500


##############
# Processing #
##############
# Load corpus
a = Author(infile)
print('Corpus characters:', len(a.corpus))

# Preprocessing (sentence split, tokenizer)
a.preprocess(Tokenizer())
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))
print('Corpus vocabulary:', len(a.parsed.vocabulary))

# Document partitioning
a.partition_into_docs(part_size)
print('Documents:', len(a.docs))
print('Document tokens:', a.docs[0].size)


# Author embedding model
m = EmbeddingModel()
m.train(a.sentences)
print('Embedding vocabulary:', len(m.vocabulary))
print('Embedding matrix:', m.vectors.shape)
