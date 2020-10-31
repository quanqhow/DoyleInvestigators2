#! /usr/bin/python3

import os
from authordetect import Author, Tokenizer, TextSpan, textutils, trainutils
from smarttimers import SmartTimer


######################
# User Configuration #
######################
infile = '../data/Doyle.txt'
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
t = SmartTimer('10/90 Split')

t.tic('Load corpus')
a = Author(infile)
t.toc()
print('Corpus characters:', len(a.corpus))

t.tic('Preprocessing: Tokenizer')
a.preprocess(Tokenizer(lemmatizer=None))
t.toc()
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))

t.tic('Document partitioning')
a.partition_into_docs(part_size, remain_factor)
t.toc()
print('Documents:', len(a.docs))
print('Document tokens:', a.docs[0].size)

t.tic('Train/test splits')
train_docs, test_docs = trainutils.split_data_into_train_test(
    a.docs,
    test_size=test_size,
    random_state=random_state,
)
train_docs = TextSpan(train_docs)
test_docs = TextSpan(test_docs)
t.toc()
print('Training documents:', len(train_docs))
print('Training tokens:', train_docs[0].size)
print('Testing documents:', len(test_docs))
print('Testing tokens:', test_docs[0].size)

# Create output directory
# Use a temporary directory to prevent overwrites, user is responsible for
# moving output to a valid destination.
if outdir and not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

t.tic('Save to file')
# Save training data to file
train_spans = [doc.span for doc in train_docs]
train_text = ' '.join(textutils.get_text_from_span(a.corpus, train_spans))
textutils.save_text(train_text, os.path.join(outdir, train_outfile))

# Save training data to file
test_spans = [doc.span for doc in test_docs]
test_text = ' '.join(textutils.get_text_from_span(a.corpus, test_spans))
textutils.save_text(test_text, os.path.join(outdir, test_outfile))
t.toc()

print('Walltime:', t.walltime)
print(t)
