#! /usr/bin/python3

import os
from authordetect import Tokenizer, load_pickle
from predict_classifier import predict
from writer2vec import writer2vec, flatten


######################
# User Configuration #
######################
seed = 0  # int, None
tokenizer = Tokenizer(min_token_length=1, use_stopwords=False)
stopwords = Tokenizer.STOPWORDS
mlp_file = 'mlp_50dim_350part.pkl'

threads = 4 if seed is None else 1
if seed is not None:
    os.environ['PYTHONHASHSEED'] = str(seed)


# Single document
test_data = '../data/Doyle_10.txt'
test_label = 1


writer2vec_params = {
    # Preprocess
    'tokenizer': tokenizer,

    # word2vec - Parameters passed directly to gensim.models.Word2Vec
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    'size': 50,
    'window': 5,
    'min_count': 1,
    'workers': threads,
    'sg': 0,
    'hs': 0,
    'negative': 20,
    'alpha': 0.03,
    'min_alpha': 0.0007,
    'seed': seed,
    'sample': 6e-5,
    'iter': 10,

    # doc2vec
    'stopwords': stopwords,  # iterable[str], None
    'use_norm': True,  # bool
}


##############
# Processing #
##############
# Document vectors and labels
vectors, labels = writer2vec(test_data, test_label, **writer2vec_params)

# Flatten data
test_vectors = flatten(vectors)
test_labels = flatten(labels)

# Predict
predict_labels, classes, probabilities, metrics = predict(mlp_file, test_vectors, test_labels)
print('True:', test_labels)
print('Predict:', predict_labels)
print('Classes:', classes)
print('Probabilities:', probabilities)
print(metrics)
