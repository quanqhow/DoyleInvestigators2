#! /usr/bin/python3

import numpy
from .textutils import load_text
from .textspan import TextSpan
from .embedding import EmbeddingModel
from .tokenizer import Tokenizer
from typing import Union, Iterable, Callable


__all__ = ['Author']


def np_avg(data: numpy.ndarray):
    return numpy.average(data, axis=0)


def np_sum(data: numpy.ndarray):
    return numpy.sum(data, axis=0)


class Author:
    def __init__(self, corpus: str = None):
        self._corpus = load_text(corpus) if corpus else corpus
        # Print info on how is input considered so that if a filename does
        # not exists, then user can be informed.
        if self._corpus == corpus:
            print('Input Mode: Author corpus was provided as raw text')
        else:
            print('Input Mode: Author corpus is loaded from a text file')
        self._parsed = TextSpan()
        self._docs = TextSpan()
        self._model = None  # EmbeddingModel
        self._docs_vectors = numpy.array([])

    @property
    def corpus(self):
        return self._corpus

    @property
    def parsed(self):
        return self._parsed

    @property
    def words(self):
        depth = self._parsed.depth
        if depth >= 2:
            tokens = list(self._parsed.iter_tokens(depth - 1))
            return TextSpan(tokens, (tokens[0].span[0], tokens[-1].span[1]))
        return TextSpan()

    @property
    def sentences(self):
        depth = self._parsed.depth
        if depth >= 3:
            tokens = list(self._parsed.iter_tokens(depth - 2))
            return TextSpan(tokens, (tokens[0].span[0], tokens[-1].span[1]))
        return TextSpan()

    @property
    def docs(self):
        return self._docs

    # NOTE: *_* properties are utility methods useful to get lists of strings.
    @property
    def words_str(self):
        return list(str(w) for w in self.words)

    @property
    def sentences_str(self):
        return list(str(s) for s in self.sentences)

    @property
    def sentences_words_str(self):
        return list(list(s.iter_tokens()) for s in self.sentences)

    @property
    def model(self):
        return self._model

    @property
    def docs_vectors(self):
        return self._docs_vectors

    def preprocess(self, tokenizer: Tokenizer = None):
        # Reset because parsed corpus might have changed
        self._docs = TextSpan()

        if tokenizer is None:
            # NOTE: No tokenizer, then represent corpus as one sentence
            # with one token.
            span = (0, len(self._corpus))
            token = TextSpan(self._corpus, span)
            sent = TextSpan([token], span)
            sents = TextSpan([sent], span)
            self._parsed = sents
        else:
            sents = TextSpan()
            for sb, se, s in tokenizer.sentencize(self._corpus):
                sent = TextSpan()
                for tb, te, t in tokenizer.tokenize(s):
                    tspan = (tb + sb, te + sb)
                    t = tokenizer.lemmatize(t)
                    sent.append(TextSpan(t, tspan))
                if len(sent) > 0:
                    sent.span = (sb, se)
                    sents.append(sent)
            if len(sents) > 0:
                sents.span = (sents[0].span[0], sents[-1].span[1])
            self._parsed = sents


    def partition_into_docs(self, size: int = 350, remain_factor: float = 1.):
        """Partition text into documents of a specified token count."""
        def partition(size, remain_factor):
            # Limit lower bound of size
            size = max(1, size)

            # Iterate through sentences
            cnt = 0
            doc = TextSpan()
            for s in self.sentences:
                cnt += len(s)
                if cnt <= size:
                    # Add sentence to current document until partition
                    # size is satisfied
                    doc.append(s)
                    if cnt < size:
                        continue
                else:
                    # Truncate last sentence for current document
                    # NOTE: Span is not truncated because it represents the
                    # actual sentences represented.
                    span = (s.span[0], s.span[1])
                    # span = (s.span[0], s[size - cnt - 1].span[1])
                    doc.append(TextSpan(s[:size - cnt], span))

                doc.span = (doc[0].span[0], doc[-1].span[1])
                yield doc

                # Reset document controls
                cnt = 0
                doc = TextSpan()

            # Consider remaining string as a document if it is "long" enough
            if doc and doc.size >= numpy.ceil(size * remain_factor):
                doc.span = (doc[0].span[0], doc[-1].span[1])
                yield doc

        docs = TextSpan(list(partition(size, remain_factor)))
        if len(docs) > 0:
            docs.span = (docs[0].span[0], docs[-1].span[1])
        self._docs = docs

    def embed(self, **kwargs):
        # Reset document embeddings
        self._docs_vectors = numpy.array([])

        self._model = EmbeddingModel(**kwargs)
        self._model.train(self.sentences_words_str)

    def embed_docs(self, **kwargs):
        # NOTE: Auto-embed with default parameters
        if self._model is None:
            self.embed()

        self._docs_vectors = numpy.array([
            type(self).doc2vec(doc, self._model, **kwargs)
            for doc in self.docs
        ])

    def writer2vec(self, **kwargs):
        """Pipeline for generating Author and document embeddings."""
        # NOTE: Ensure that parameter names do not collide.
        self.preprocess(kwargs.pop('tokenizer', None))
        self.partition_into_docs(
            size=kwargs.pop('part_size', 350),
            remain_factor=kwargs.pop('remain_factor', 1.),
        )
        # Extract arguments for operations after embed(), because it consumes
        # remaining kwargs.
        stopwords = kwargs.pop('stopwords', None)
        func = kwargs.pop('func', np_avg)
        use_norm = kwargs.pop('use_norm', False)
        self.embed(**kwargs)
        self.embed_docs(
            stopwords=stopwords,
            func=func,
            use_norm=use_norm,
        )

    def save(self, author_file, doc_file):
        """Save Author's state: author embedding, document embedding, and
        parameters."""
        pass

    def load(self, author_file, doc_file):
        pass

    @staticmethod
    def doc2vec(
        doc: 'TextSpan',
        model: Union['gensim.models.word2vec', EmbeddingModel],
        *,
        stopwords: Iterable[str] = None,
        func: Callable = np_avg,
        use_norm: bool = False,
    ):
        if stopwords is None:
            stopwords = set()

        if isinstance(model, EmbeddingModel):
            model = model.model

        return func(numpy.array([
            model.wv.word_vec(word, use_norm)
            for word in doc.tokens
            if word not in stopwords
        ]))
