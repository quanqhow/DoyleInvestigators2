#! /usr/bin/python3

from authordetect import Tokenizer, save_pickle, np_avg, np_sum
from train_classifier import train_classifier
from writer2vec import writer2vec, split_combine_data, flatten


######################
# User Configuration #
######################
seed = 0  # int, None
# mlp_file = 'mlp.pkl'
mlp_file = None
# embedding_file = 'doyle.bin'
embedding_file = None


# Positive author first
train_data = [
    '../data/Doyle_90.txt',
    '../data/Rinehart_90.txt',
    '../data/Christie_90.txt',
]
train_labels = [1, 0, 0]


writer2vec_params = {
    # Document partitioning
    'part_size': 350,  # int, None
    # 'remain_factor': 350/350,  # float [0,1], default=1

    # word2vec - Parameters passed directly to gensim.models.Word2Vec
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    'size': 300,
    'window': 5,
    'min_count': 1,
    'sg': 0,
    'hs': 0,
    'negative': 20,
    'alpha': 0.03,
    'min_alpha': 0.0007,
    'seed': seed,
    'sample': 6e-5,
    'iter': 10,

    # doc2vec
    'stopwords': Tokenizer.STOPWORDS,  # iterable[str], None
    'func': np_avg,  # callable
    'use_norm': True,  # bool
    'missing_value': 0,  # int
}

# Train/test data split - Parameters passed directly to sklearn.model_selection.train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
train_test_params = {
    'test_size': 0.1,  # train_size=1-test_size
    'random_state': seed,
}

# Classifier - Paramaters passed directly to sklearn.neural_network.MLPClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
mlp_params = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'lbfgs',
    'alpha': 1e-5,
    'random_state': seed,
    'learning_rate': 'constant',  # only used when solver='sgd'
    'max_iter': 200,
    'shuffle': True,  # only used when solver='sgd' or 'adam'
    'warm_start': False,
    'momentum': 0.9,  # only used when solver='sgd'
    'max_fun': 15000,  # only used when solver='lbfgs'
}


##############
# Processing #
##############
# Document vectors and labels
vectors, labels = writer2vec(train_data, train_labels, outfiles=embedding_file, **writer2vec_params)

# Fraction select (50% of 90% for positive, 25% of 90% for each negative)
vectors, labels = split_combine_data(vectors, labels, seed=seed)

# Flatten data
train_vectors = flatten(vectors)
train_labels = flatten(labels)

# Train
mlp = train_classifier(train_vectors, train_labels, train_test_params, **mlp_params)

# Save classifier model
if mlp_file:
    save_pickle(mlp, mlp_file)
