#! /usr/bin/python3

import pickle


def load_pickle(fn: str):
    with open(fn, 'rb') as fd:
        return pickle.load(fd)


vectors_file = 'vectors.pickle'
vectors_norm_file = 'vectors_norm.pickle'
mlp_file = 'mlp.pickle'

# Load document vectors and corresponding labels
doc_vectors, labels = load_pickle(vectors_file)
doc_vectors_norm, labels = load_pickle(vectors_norm_file)
print(doc_vectors.shape)
print(doc_vectors_norm.shape)
print(len(labels))

# Load MLP classifier model
mlp = load_pickle(mlp_file)
print(mlp.coefs_[0].shape)
print(mlp.coefs_[1].shape)
