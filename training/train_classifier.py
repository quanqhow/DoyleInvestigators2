#! /usr/bin/python3

import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from typing import Any, Dict, Iterable


def train_classifier(
    vectors: Iterable[Iterable[float]],
    labels: Iterable[int],
    train_test_params: Dict[str, Any] = None,
    **kwargs,
) -> 'MLPClassifier':
    if train_test_params is None:
        train_test_params = {}

    vectors = numpy.array(vectors)
    labels = numpy.array(labels)
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(
        vectors, labels,
        stratify=labels,
        **train_test_params,
    )

    mlp = MLPClassifier(**kwargs)
    mlp.fit(train_vectors, train_labels)

    return mlp
