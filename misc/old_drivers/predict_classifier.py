#! /usr/bin/python3

import os
import sys
import numpy
import random
import functools
import itertools
from authordetect import Author, Tokenizer, SmartTimer, load_pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Any, Dict, Tuple, Union, Iterable

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
verbose = True
seed = 0  # int, None
tokenizer = Tokenizer(min_token_length=1, use_stopwords=False)
stopwords = Tokenizer.STOPWORDS
mlp = 'mlp.pkl'


test_data = [
    '../data/Doyle_10.txt',
    '../data/Rinehart_10.txt',
    '../data/Christie_10.txt',
]
test_labels = [1, 0, 0]


writer2vec_params = [
    {
        'verbose': verbose,

        # Preprocess
        'tokenizer': tokenizer,

        # Document partitioning
        'part_size': 350,  # int, None

        # word2vec - Parameters passed directly to gensim.models.Word2Vec
        # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        'size': 50,
        'window': 5,
        'min_count': 1,
        'workers': 4,
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
    },
]


def writer2vec(
    data: Iterable[str],
    labels: Iterable[int],
    *,
    verbose: bool = True,
    **kwargs,
):
    # Verbosity
    fd = sys.stdout if verbose else open(os.devnull, 'w')
    vprint = functools.partial(print, file=fd)

    t = SmartTimer('Document Embedding Pipeline')
    authors = []
    for i, (corpus, label) in enumerate(zip(data, labels), start=1):
        t.tic(f'{i}: Load corpus')
        author = Author(corpus, label)
        authors.append(author)
        t.toc()
        vprint(f'Author {i}:')
        vprint('\tCharacter count:', len(author.corpus))

        t.tic(f'{i}: writer2vec')
        author.writer2vec(**kwargs)
        t.toc()
        vprint('\tCorpus sentences:', len(author.sentences))
        vprint('\tCorpus tokens:', len(author.words))
        vprint('\tCorpus vocabulary:', len(author.parsed.vocabulary))
        vprint('\tDocuments:', len(author.docs))
        vprint('\tDocument tokens:', author.docs[0].size)
        vprint('\tEmbedding vocabulary:', len(author.embedding.vocabulary))
        vprint('\tEmbedding matrix:', author.embedding.vectors.shape)
        vprint('\tDocuments embedding matrix:', author.docs_vectors.shape)

    vprint('Splitting train/test documents...')
    t.tic('Train/test documents split')
    vectors = []
    labels = []
    for author in authors:
        vectors.append(author.docs_vectors)
        labels.append([author.label] * len(author.docs_vectors))
    t.toc()

    vprint('writer2vec pipeline walltime (s):', t.walltime)
    vprint(t)

    return vectors, labels


def predict(
    mlp: Union[str, 'MLPClassifier'],
    vectors: Iterable[Iterable[float]],
    true_labels: Iterable[int] = None,
    *,
    verbose: bool = True,
):
    # Verbosity
    fd = sys.stdout if verbose else open(os.devnull, 'w')
    vprint = functools.partial(print, file=fd)

    t = SmartTimer('Prediction Classifier Pipeline')
    vprint('Starting prediction classifier pipeline...')
    vprint('Data items:', (len(vectors), true_labels))

    if isinstance(mlp, str):
        vprint(f'Loading classifier model from {mlp}')
        t.tic('Load classifier model')
        mlp = load_pickle(mlp)
        t.toc()

    t.tic('MLP Prediction')
    predict_labels = mlp.predict(vectors)
    probabilities = mlp.predict_proba(vectors)
    vprint('Predictions:', predict_labels)
    vprint('Probabilities:', probabilities)

    if true_labels is not None:
        score = mlp.score(vectors, true_labels)
        f1 = f1_score(true_labels, predict_labels, zero_division=1)
        precision, recall, fbeta, support = precision_recall_fscore_support(
            true_labels, predict_labels, zero_division=1
        )
        vprint('True labels:', true_labels)
        vprint('Score:', score)
        vprint('F1 score:', f1)
        vprint('Precision:', precision)
        vprint('Recall:', recall)
        vprint('F-beta score:', fbeta)
        vprint('Support:', support)
        metrics = (f1, precision, recall)
    else:
        metrics = None
    t.toc()

    vprint('Classification pipeline walltime (s):', t.walltime)
    vprint(t)

    return predict_labels, probabilities, metrics


def split_combine_data(data, labels, pos_frac=0.5, neg_frac=None, *, seed=None):
    """Split a nested dataset of the form [[pos], [neg], [neg], ...],
    using the fractions provided.

    The positive data is first and is used completely with the pos_frac
    stating the fraction it represents. Partial negative datasets are
    generated via random sampling. Data is returned in the same format.
    """
    random.seed(seed)

    total_size = len(data[0]) / pos_frac

    # Calculate negative fraction, uniformly for each negative dataset
    _neg_frac = (1 - pos_frac) / (len(data) - 1)
    if neg_frac is not None:
        _neg_frac = min(_neg_frac, neg_frac)
    neg_frac = _neg_frac

    # Randomly select negative data
    combined_data = [data[0]]
    combined_labels = [labels[0]]
    neg_size = int(total_size * neg_frac)
    for neg_data, neg_labels in zip(data[1:], labels[1:]):
        _neg_size = min(neg_size, len(neg_data))
        idxs = random.sample(range(len(neg_data)), _neg_size)
        _neg_data = [neg_data[i] for i in idxs]
        _neg_labels = [neg_labels[i] for i in idxs]
        combined_data.append(_neg_data)
        combined_labels.append(_neg_labels)

    return combined_data, combined_labels


if __name__ == '__main__':
    # Check if input data is text or vectors
    if isinstance(test_data[0], str):
        test_vectors, test_labels = writer2vec(test_data, test_labels, **writer2vec_params[0])

    # Fraction select and flatten
    frac_test_vectors, frac_test_labels = split_combine_data(test_vectors, test_labels, seed=seed)
    test_vectors = numpy.array(list(itertools.chain.from_iterable(frac_test_vectors)))
    test_labels = numpy.array(list(itertools.chain.from_iterable(frac_test_labels)))

    # Predict
    true_labels, metrics = predict(mlp, test_vectors, test_labels)
