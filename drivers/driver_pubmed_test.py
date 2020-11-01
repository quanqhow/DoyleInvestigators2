#! /usr/bin/python3

import numpy
from authordetect import Author, Tokenizer, SmartTimer, load_pickle
)

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
# import os
# os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
infile = '../data/Doyle_90.txt'
part_size = 350
workers = 4
seed = None # 0


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
author = Author(doyle_infile)
t.toc()
print('Corpus characters:', len(author.corpus))

t.tic(f'writer2vec')
author.writer2vec(
    tokenizer=Tokenizer(),
    stopwords=Tokenizer.STOPWORDS,
    part_size=part_size,
    workers=workers,
    seed=seed,
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
test_labels = [0] * len(test_vectors)
print('Test vectors:', len(test_vectors))
print('Test labels:', len(test_labels))

t.tic('Load MLP classifier')
mlp = load_pickle('mlp.pkl')
t.toc()

t.tic('MLP Prediction')
predictions = mlp.predict(test_vectors)
score = mlp.score(test_vectors, test_labels)
t.toc()
print('Predictions:', predictions)
print('Test labels:', test_labels)
print('Score:', score)

print('Walltime:', t.walltime)
