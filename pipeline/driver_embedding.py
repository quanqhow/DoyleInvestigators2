#! /usr/bin/python3

import os
from author import Author, Tokenizer
from train import split_data_into_train_test
from textspan import TextSpan
import textutils
import embedding


######################
# User Configuration #
######################
infile = '../data/Doyle_90.txt'
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
a = Author(infile)
print('Corpus characters:', len(a.corpus))

# Preprocessing (sentence split, tokenizer)
tokenizer = Tokenizer(lemmatizer=None)
a.preprocess(tokenizer)
print('Corpus sentences:', len(list(a.sents)))
print('Corpus tokens:', len(list(a.words)))

# Document partitioning
a.partition_into_docs(part_size, remain_factor)
print('Documents:', len(a.docs))
print('Document tokens:', a.docs[0].size)

# Document splits (train, test)
train_docs, test_docs = split_data_into_train_test(
    a.docs,
    test_size=test_size,
    random_state=random_state,
)
train_docs = TextSpan(train_docs)
test_docs = TextSpan(test_docs)
print('Training documents:', len(train_docs))
print('Training tokens:', train_docs[0].size)
print('Testing documents:', len(test_docs))
print('Testing tokens:', test_docs[0].size)


# Create output directory
# Use a temporary directory to prevent overwrites, user is responsible for
# moving output to final destination.
if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

# Save training data to file
train_spans = [doc.span for doc in train_docs]
train_text = ' '.join(textutils.get_text_from_span(a.corpus, train_spans))
textutils.save_text(train_text, os.path.join(outdir, train_outfile))

# Save training data to file
test_spans = [doc.span for doc in test_docs]
test_text = ' '.join(textutils.get_text_from_span(a.corpus, test_spans))
textutils.save_text(test_text, os.path.join(outdir, test_outfile))
