#! /usr/bin/python3

from .textutils import load_pickle, save_pickle
from sklearn.neural_network import MLPClassifier
from typing import Any, Iterable


__all__ = ['Classifier']


class Classifier:
    def __init__(self, classifier_file: str = None, **kwargs):
        if classifier_file:
            self._clf = type(self).load(classifier_file)
        else:
            self._clf = MLPClassifier(
                solver=kwargs.pop('solver', 'lbfgs'),
                alpha=kwargs.pop('alpha', 1e-5),
                hidden_layer_sizes=kwargs.pop('hidden_layer_sizes', (100,)),
                random_state=kwargs.pop('random_state', None),
                **kwargs,
            )

    @property
    def model(self):
        return self._clf

    def train(self, features: Iterable[Any], labels: Iterable[Any], **kwargs):
        self._clf.fit(features, labels, **kwargs)

    def predict(self, features: Iterable[Any], **kwargs):
        return self._clf.predict(features, **kwargs)

    def save(self, outfile: str):
        save_pickle(self._clf, outfile)

    @staticmethod
    def load(fn: str):
        return load_pickle(fn)
