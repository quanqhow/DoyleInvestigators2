#! /usr/bin/python3

import numpy
from authordetect import Author, Tokenizer, SmartTimer, save_pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
# import os
# os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
doyle_infile = '../data/Doyle_90.txt'
rinehart_infile = '../data/Rinehart_90.txt'
christie_infile = '../data/Christie_90.txt'
part_size = 350
workers = 4
seed = None # 0
test_size = 0.1

train_test_params = {
    'test_size': test_size,
    'random_state': seed,
}

mlp_params = {
    'solver': 'lbfgs',
    'alpha': 1e-5,
    'hidden_layer_sizes': (50,),
    'random_state': seed,
}


##############
# Processing #
##############
t = SmartTimer('Pipeline')

t.tic('Load corpus')
doyle = Author(doyle_infile)
rinehart = Author(rinehart_infile)
christie = Author(christie_infile)
t.toc()
print('Doyle corpus characters:', len(doyle.corpus))
print('Rinehart corpus characters:', len(rinehart.corpus))
print('Christie corpus characters:', len(christie.corpus))

# Names and object handles to enable looping through same operations
names = ['Doyle', 'Rinehart', 'Christie']
authors = [doyle, rinehart, christie]

for name, author in zip(names, authors):
    t.tic(f'{name}: writer2vec')
    author.writer2vec(
        tokenizer=Tokenizer(),
        stopwords=Tokenizer.STOPWORDS,
        part_size=part_size,
        workers=workers,
        seed=seed,
        use_norm=True,
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

# Create train/test vectors/labels
t.tic('Train/test data prep')
label_true = [1] * len(doyle.docs_vectors)
label_false = [0] * (len(rinehart.docs_vectors) + len(christie.docs_vectors))
labels = numpy.array([*label_true, *label_false])
vectors = numpy.vstack([doyle.docs_vectors, rinehart.docs_vectors, christie.docs_vectors])
train_vectors, test_vectors, train_labels, test_labels = train_test_split(
    vectors, labels,
    stratify=labels,
    **train_test_params,
)
t.toc()
print('Train vectors:', len(train_vectors))
print('Train labels:', len(train_labels))
print('Test vectors:', len(test_vectors))
print('Test labels:', len(test_labels))

t.tic('Training MLP classifier')
mlp = MLPClassifier(**mlp_params)
mlp.fit(train_vectors, train_labels)
t.toc()

t.tic('Save MLP classifier')
# save_pickle(mlp, 'mlp.pkl')
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
