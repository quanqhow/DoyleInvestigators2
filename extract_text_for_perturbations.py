#! /usr/bin/python3

from functions import *


# Load corpus
corpus = load_corpus('data/Doyle.txt')
print('Corpus characters:', len(corpus))

# Preprocessing (sentence split, tokenizer, lemmatizer)
sents, sents_spans = tokenize(corpus, lemmatizer=None, with_spans=True)
print('Corpus sentences:', len(sents))
print('Corpus tokens:', len(list(flatten(sents))))

# Document partitioning
docs, docs_spans = partition(sents, spans=sents_spans)
print('Documents:', len(docs))
print('Document tokens:', token_count(docs[0]))

# Document splits (train, test)
train_docs, test_docs = docs_split(docs, train_size=0.9, random_state=0)
train_idxs, test_idxs = next(shuffle_split(docs, train_size=0.9, random_state=0))
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

# Save training data to file
train_text = ' '.join(get_documents_text_by_spans(corpus, train_spans))
save_corpus(train_text, 'data/Doyle_90.txt')

# Save testing data to file
test_text = ' '.join(get_documents_text_by_spans(corpus, test_spans))
save_corpus(test_text, 'data/Doyle_10.txt')
