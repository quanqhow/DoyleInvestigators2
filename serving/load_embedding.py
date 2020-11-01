#! /usr/bin/python3

import pickle


def load_pickle(fn: str):
    with open(fn, 'rb') as fd:
        return pickle.load(fd)


# Load MLP classifier model
mlp = load_pickle('mlp_50dim_350part.pkl')
print(mlp.coefs_[0].shape)
print(mlp.coefs_[1].shape)
