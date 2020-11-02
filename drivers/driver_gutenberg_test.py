#! /usr/bin/python3

import numpy
from authordetect import Author, Tokenizer, SmartTimer, load_pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
# import os
# os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
# infile = 'https://www.gutenberg.org/files/244/244-0.txt'
# label = 1
infile = 'https://www.gutenberg.org/files/863/863-0.txt'
label = 0
part_size = 350
workers = 4
seed = None # 0


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
author = Author(infile)
t.toc()
print('Corpus characters:', len(author.corpus))

t.tic(f'writer2vec')
author.writer2vec(
    tokenizer=Tokenizer(),
    stopwords=Tokenizer.STOPWORDS,
    part_size=part_size,
    workers=workers,
    seed=seed,
    use_norm=True,
)
t.toc()

print('Corpus sentences:', len(author.sentences))
print('Corpus tokens:', len(author.words))
print('Corpus vocabulary:', len(author.parsed.vocabulary))
print('Documents:', len(author.docs))
print('Document tokens:', author.docs[0].size)
print('Embedding vocabulary:', len(author.model.vocabulary))
print('Embedding matrix:', author.model.vectors.shape)
print('Documents embedding matrix:', author.docs_vectors.shape)

test_vectors = author.docs_vectors
test_labels = numpy.array([label] * len(test_vectors))
print('Test vectors:', len(test_vectors))
print('Test labels:', len(test_labels))

t.tic('Load MLP classifier')
mlp = load_pickle('mlp.pkl')
t.toc()

t.tic('MLP Prediction')
predict_labels = mlp.predict(test_vectors)
probabilities = mlp.predict_proba(test_vectors)
score = mlp.score(test_vectors, test_labels)
f1 = f1_score(test_labels, predict_labels, zero_division=1)
precision, recall, fbeta, support = precision_recall_fscore_support(
    test_labels, predict_labels, zero_division=1
)
t.toc()
print('Predictions:', predict_labels)
print('Test labels:', test_labels)
print('Probabilities:', probabilities)
print('Score:', score)
print('F1 score:', f1)
print('Precision:', precision)
print('Recall:', recall)
print('F-beta score:', fbeta)
print('Support:', support)

print('Walltime:', t.walltime)
print(t)
