#! /usr/bin/python3

import os
# import multiprocessing
import psutil
from gensim.models import Word2Vec
from typing import Iterable


class EmbeddingModel:
    def __init__(self, **kwargs):
        self._sentences = None
        self._params = None
        self._model = Word2Vec(
            size=kwargs.pop('size', 50),
            window=kwargs.pop('window', 5),
            min_count=kwargs.pop('min_count', 1),
            # workers=kwargs.pop('workers', multiprocessing.cpu_count() / 2),
            workers=kwargs.pop('workers', psutil.cpu_count(False)),
            sg=kwargs.pop('sg', 0),
            hs=kwargs.pop('hs', 0),
            negative=kwargs.pop('negative', 20),
            alpha=kwargs.pop('alpha', 0.03),
            min_alpha=kwargs.pop('min_alpha', 0.0007),
            seed=kwargs.pop('seed', 0),
            max_vocab_size=kwargs.pop('max_vocab_size', None),
            max_final_vocab=kwargs.pop('max_final_vocab', None),
            sample=kwargs.pop('sample', 6e-5),
            iter=kwargs.pop('iter', 10),
            sorted_vocab=kwargs.pop('sorted_vocab', 1),
            **kwargs,
        )

    @property
    def model(self):
        return self._model

    @property
    def sentences(self):
        return self._sentences

    @property
    def vocabulary(self):
        return self._model.wv.vocab

    @property
    def vectors(self):
        return self._model.wv.vectors

    @property
    def vectors_norm(self):
        self._model.wv.init_sims()
        return self._model.wv.vectors_norm

    def train(self, sentences: Iterable[Iterable[str]] = None, **kwargs):
        # Build vocabulary table
        self._model.build_vocab(sentences, progress_per=10000)

        # Training of the model
        self._model.train(
            sentences,
            total_examples=kwargs.pop('total_examples', self._model.corpus_count),
            epochs=kwargs.pop('epochs', self._model.iter),
            report_delay=kwargs.pop('report_delay', 1),
            **kwargs,
        )

        # NOTE: Sentences are not deepcopied, so they are prone to side-effects
        self._sentences = sentences

    def save(self, outfile):
        # Create output directory (if available)
        outdir = os.path.dirname(outfile)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)

        # Save the model
        if outfile is not None:
            # self._model.save_word2vec_format(outfile)
            self._model.save(outfile)

    def load(self, infile):
        self._sentences = None
        self._model.load(infile)
