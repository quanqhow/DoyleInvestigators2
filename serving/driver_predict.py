#! /usr/bin/python3

from authordetect import Tokenizer, EmbeddingModel, load_pickle, np_avg, np_sum
from predict_classifier import predict
from writer2vec import writer2vec, flatten


######################
# User Configuration #
######################
mlp_file = 'mlp.pkl'
embedding_file = 'doyle.bin'

test_data = '../data/Doyle_90.txt'
test_label = 0

writer2vec_params = {
    'part_size': 350,

    # doc2vec
    'stopwords': Tokenizer.STOPWORDS,  # iterable[str], None
    'func': np_avg,  # callable
    'use_norm': True,  # bool
    'missing_value': 0,  # int
}


##############
# Processing #
##############
# Load embedding model
embedding = EmbeddingModel()
embedding.load(embedding_file)

# Document vectors and labels
vectors, labels = writer2vec(test_data, test_label, embedding=embedding, **writer2vec_params)

# Flatten data
test_vectors = flatten(vectors)
test_labels = flatten(labels)

# Load classifier model
mlp = load_pickle(mlp_file)

# Predict
predict_labels, classes, probabilities, metrics = predict(mlp, test_vectors, test_labels)
print('True:', test_labels)
print('Predict:', predict_labels)
print('Classes:', classes)
print('Probabilities:', probabilities)
print(metrics)
