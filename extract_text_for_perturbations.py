#! /usr/bin/python3

import os
from functions import *


######################
# User Configuration #
######################
infile = 'data/Doyle.txt'
outdir = 'tmp'
random_state = 0

part_size = 3500
# Factor for capturing as much as possible from trailing text
# Default is 1. but set to 0.1 because 3500/350=10%
remain_factor = 0.1

test_size = 0.1
train_outfile = 'Doyle_90.txt'
test_outfile = 'Doyle_10.txt'


##############
# Processing #
##############
# Load corpus
corpus = load_corpus(infile)
print('Corpus characters:', len(corpus))

# Preprocessing (sentence split, tokenizer)
sents, sents_spans = tokenize(corpus, lemmatizer=None, with_spans=True)
print('Corpus sentences:', len(sents))
print('Corpus tokens:', len(list(flatten(sents))))

# Document partitioning
docs, docs_spans = partition(
    sents, size=part_size, remain_factor=remain_factor, spans=sents_spans,
)
print('Documents:', len(docs))
print('Document tokens:', token_count(docs[0]))

# Document splits (train, test)
train_docs, test_docs = docs_split(
    docs, test_size=test_size, random_state=random_state
)
train_idxs, test_idxs = next(shuffle_split(
    docs, test_size=test_size, random_state=random_state
))
train_spans = [docs_spans[i] for i in train_idxs]
test_spans = [docs_spans[i] for i in test_idxs]
print('Training documents:', len(train_docs))
print('Training indexes:', len(train_idxs))
print('Training spans:', len(train_spans))
print('Training tokens:', token_count(train_docs[0]))
print('Testing documents:', len(test_docs))
print('Testing indexes:', len(test_idxs))
print('Testing spans:', len(test_spans))
print('Testing tokens:', token_count(test_docs[0]))

# Create output directory
# Use a temporary directory to prevent overwrites, user is responsible for
# moving output to final destination.
if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

# Save training data to file
train_text = ' '.join(get_documents_text_by_spans(corpus, train_spans))
save_corpus(train_text, os.path.join(outdir, train_outfile))

# Save testing data to file
test_text = ' '.join(get_documents_text_by_spans(corpus, test_spans))
save_corpus(test_text, os.path.join(outdir, test_outfile))
