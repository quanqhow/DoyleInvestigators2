#! /usr/bin/python3

from authordetect import Author, Tokenizer, SmartTimer


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
a.embed(workers=workers, seed=seed)
t.toc()
print('Embedding vocabulary:', len(a.model.vocabulary))
print('Embedding matrix:', a.model.vectors.shape)

t.tic('Documents embedding')
a.embed_docs(stopwords=Tokenizer.STOPWORDS)
t.toc()
print('Documents embedding matrix:', a.docs_vectors.shape)

t.tic('Save author embedding')
# a.model.save(author_embedfile)
t.toc()

print('Walltime:', t.walltime)
print(t)
