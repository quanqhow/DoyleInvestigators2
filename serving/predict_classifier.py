#! /usr/bin/python3

from authordetect import load_pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Any, Union, Iterable


def predict(
    mlp: Union[str, 'MLPClassifier'],
    vectors: Iterable[Iterable[float]],
    true_labels: Iterable[Any] = None,
):
    if isinstance(mlp, str):
        mlp = load_pickle(mlp)

    predict_labels = mlp.predict(vectors)
    probabilities = mlp.predict_proba(vectors)

    if true_labels is not None:
        score = mlp.score(vectors, true_labels)
        f1 = f1_score(true_labels, predict_labels, zero_division=1)
        precision, recall, fbeta, support = precision_recall_fscore_support(
            true_labels, predict_labels, zero_division=1
        )
        metrics = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
        }
    else:
        metrics = None

    return predict_labels, mlp.classes_, probabilities, metrics
