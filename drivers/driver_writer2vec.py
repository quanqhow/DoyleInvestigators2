#! /usr/bin/python3

from authordetect import Author, Tokenizer, SmartTimer

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
import os
os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
infile = '../data/Doyle_10.txt'
author_embedfile = 'model.bin'
part_size = 3500
workers = 1
seed = 0


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
a = Author(infile)
t.toc()
print('Corpus characters:', len(a.corpus))

t.tic('writer2vec')
a.writer2vec(
    tokenizer=Tokenizer(),
    part_size=part_size,
    workers=workers,
    seed=seed,
)
t.toc()
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))
print('Corpus vocabulary:', len(a.parsed.vocabulary))
print('Documents:', len(a.docs))
print('Document tokens:', a.docs[0].size)
print('Embedding vocabulary:', len(a.model.vocabulary))
print('Embedding matrix:', a.model.vectors.shape)
print('Documents embedding matrix:', a.docs_vectors.shape)

t.tic('Save author embedding')
a.model.save(author_embedfile)
t.toc()

print('Walltime:', t.walltime)
print(t)
