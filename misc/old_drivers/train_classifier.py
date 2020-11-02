#! /usr/bin/python3

import os
import sys
import numpy
import random
import functools
import itertools
from authordetect import Author, Tokenizer, SmartTimer, save_pickle
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPClassifier
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
mlp_file = 'mlp.pkl'


train_data = [
    '../data/Doyle_90.txt',
    '../data/Rinehart_90.txt',
    '../data/Christie_90.txt',
]
train_labels = [1, 0, 0]


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

# Train/test data split - Parameters passed directly to sklearn.model_selection.train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
train_test_params = [
    {
        'test_size': 0.1,  # train_size=1-test_size
        'random_state': seed,
    },
]

# Classifier - Paramaters passed directly to sklearn.neural_network.MLPClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
mlp_params = [
    {
        'verbose': verbose,
        'outfile': mlp_file,

        'hidden_layer_sizes': (50,),
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


def train_classifier(
    vectors: Iterable[Iterable[float]],
    labels: Iterable[int],
    train_test_params: Dict[str, Any] = None,
    *,
    outfile: str = None,
    verbose: bool = True,
    **kwargs,
) -> 'MLPClassifier':
    # Verbosity
    fd = sys.stdout if verbose else open(os.devnull, 'w')
    vprint = functools.partial(print, file=fd)

    if train_test_params is None:
        train_test_params = {}

    t = SmartTimer('Training Classifier Pipeline')
    vprint('Starting training classifier pipeline...')
    vprint('Data items:', (len(vectors), len(labels)))

    vprint('Splitting train/test data')
    t.tic('Train/test split')
    vectors = numpy.array(vectors)
    labels = numpy.array(labels)
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(
        vectors, labels,
        stratify=labels,
        **train_test_params,
    )
    t.toc()
    vprint('Train vectors:', len(train_vectors))
    vprint('Train labels:', len(train_labels))
    vprint('Test vectors:', len(test_vectors))
    vprint('Test labels:', len(test_labels))

    vprint('Training MLP classifier...')
    t.tic('Train MLP classifier')
    mlp = MLPClassifier(**kwargs)
    mlp.fit(train_vectors, train_labels)
    t.toc()

    if outfile:
        vprint('Saving MLP classifier...')
        t.tic('Save MLP classifier')
        save_pickle(mlp, outfile)
        t.toc()

    vprint('Training classifier pipeline walltime (s):', t.walltime)
    vprint(t)

    return mlp


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
    if isinstance(train_data[0], str):
        train_vectors, train_labels = writer2vec(train_data, train_labels, **writer2vec_params[0])

    # Fraction select and flatten
    frac_train_vectors, frac_train_labels = split_combine_data(train_vectors, train_labels, seed=seed)
    train_vectors = numpy.array(list(itertools.chain.from_iterable(frac_train_vectors)))
    train_labels = numpy.array(list(itertools.chain.from_iterable(frac_train_labels)))

    # Train
    mlp = train_classifier(train_vectors, train_labels, train_test_params[0], **mlp_params[0])
