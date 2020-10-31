#! /usr/bin/python3

import numpy
from authordetect import Author, Tokenizer, Classifier
from smarttimers import SmartTimer

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
import os
os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
doyle_infile = '../data/Doyle_10.txt'
rinehart_infile = '../data/Rinehart_10.txt'
christie_infile = '../data/Christie_10.txt'
part_size = 350
workers = 1
seed = 0


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
doyle = Author(doyle_infile)
rinehart = Author(rinehart_infile)
christie = Author(christie_infile)
print('Doyle corpus characters:', len(doyle.corpus))
print('Rinehart corpus characters:', len(rinehart.corpus))
print('Christie corpus characters:', len(christie.corpus))
t.toc()

# Names and object handles to enable looping through same operations
names = ['Doyle', 'Rinehart', 'Christie']
authors = [doyle, rinehart, christie]

t.tic('writer2vec')
for author in authors:
    author.writer2vec(
        tokenizer=Tokenizer(),
        stopwords=Tokenizer.STOPWORDS,
        part_size=part_size,
        workers=workers,
        seed=seed,
    )
t.toc()

for name, author in zip(names, authors):
    print(f'{name} corpus sentences:', len(author.sentences))
    print(f'{name} corpus tokens:', len(author.words))
    print(f'{name} corpus vocabulary:', len(author.parsed.vocabulary))
    print(f'{name} documents:', len(author.docs))
    print(f'{name} document tokens:', author.docs[0].size)
    print(f'{name} embedding vocabulary:', len(author.model.vocabulary))
    print(f'{name} embedding matrix:', author.model.vectors.shape)
    print(f'{name} documents embedding matrix:', author.docs_vectors.shape)

t.tic('Training MLP classifier')
label_true = [1] * len(doyle.docs_vectors)
label_false = [0] * (len(rinehart.docs_vectors) + len(christie.docs_vectors))
labels = [*label_true, *label_false]

features = numpy.vstack([doyle.docs_vectors, rinehart.docs_vectors, christie.docs_vectors])

mlp = Classifier(random_state=seed)
mlp.train(features, labels)
t.toc()

t.tic('MLP Prediction')
features = [*doyle.docs_vectors[:5], *rinehart.docs_vectors[:5], *christie.docs_vectors[:5]]
predictions = mlp.predict(features)
print(predictions)
t.toc()

t.tic('Save MLP classifier')
# mlp.save('mlp.pickle')
t.toc()
