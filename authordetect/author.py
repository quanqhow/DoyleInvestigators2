#! /usr/bin/python3

import numpy
from unidecode import unidecode
from .textutils import load_text, load_pickle, save_pickle
from .textspan import TextSpan
from .embedding import EmbeddingModel
from .tokenizer import Tokenizer
from typing import Any, Union, Iterable, Callable


__all__ = ['Author']


def np_avg(data: numpy.ndarray):
    return numpy.average(data, axis=0)


def np_sum(data: numpy.ndarray):
    return numpy.sum(data, axis=0)


class Author:
    def __init__(self, text: str, label: Any = None):
        self._text = unidecode(load_text(text) if text else text)
        self._label = label
        self._parsed_text = TextSpan()
        self._parsed_documents = TextSpan()
        self._embedding = None  # EmbeddingModel
        self._dv = numpy.array([])
        self._dv_norm = numpy.array([])

    @property
    def label(self):
        return self._label

    @property
    def text(self):
        return self._text

    @property
    def words(self):
        return list(str(w) for w in self.parsed_words)

    @property
    def sentences(self):
        return list(str(s) for s in self.parsed_sentences)

    @property
    def sentences_and_words(self):
        return list(list(s.iter_tokens()) for s in self.parsed_sentences)

    @property
    def documents(self):
        return list(str(d) for d in self.parsed_documents)

    @property
    def parsed_text(self):
        return self._parsed_text

    @property
    def parsed_words(self):
        return self._parsed_text.get_tokens(2)

    @property
    def parsed_sentences(self):
        return self._parsed_text.get_tokens(3)

    @property
    def parsed_documents(self):
        return self._parsed_documents

    @property
    def embedding(self):
        return self._embedding

    @property
    def dv(self):
        return self._dv

    @property
    def dv_norm(self):
        return self._dv_norm

    def preprocess(self, tokenizer: Tokenizer = Tokenizer()):
        # Reset because parsed text might have changed
        self._parsed_documents = TextSpan()
        self._embedding = None
        self._dv = numpy.array([])
        self._dv_norm = numpy.array([])

        if tokenizer is None:
            # NOTE: No tokenizer, then represent text as one sentence
            # with one token.
            span = (0, len(self._text))
            token = TextSpan(self._text, span)
            sent = TextSpan([token], span)
            sents = TextSpan([sent], span)
            self._parsed_text = sents
        else:
            sents = TextSpan()
            for sb, se, s in tokenizer.sentencize(self._text):
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
            self._parsed_text = sents

    def partition_into_documents(self, size: int = None, remain_factor: float = 1.):
        """Partition text into documents of a specified token count."""
        def partition(size, remain_factor):
            # Limit lower bound of size
            size = max(1, size)

            # Iterate through sentences
            cnt = 0
            doc = TextSpan()
            for s in self.parsed_sentences:
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

        # If no partition size provided, then consider a single document
        if size is None or size < 1:
            docs = TextSpan([self.parsed_sentences])
        else:
            docs = TextSpan(list(partition(size, remain_factor)))

        if len(docs) > 0:
            docs.span = (docs[0].span[0], docs[-1].span[1])
        self._parsed_documents = docs

    def embed(self, embedding=None, **kwargs):
        # Reset document embeddings
        self._dv = numpy.array([])
        self._dv_norm = numpy.array([])

        if embedding is None:
            self._embedding = EmbeddingModel(**kwargs)
            self._embedding.train(self.sentences_and_words)
        elif isinstance(embedding, str):
            self._embedding = EmbeddingModel(embedding)
        else:
            self._embedding = embedding

    def embed_documents(self, embedding=None, **kwargs):
        # NOTE: Auto-embed with default parameters
        if self._embedding is None or embedding is not None:
            self.embed(embedding)

        # Use norm vectors
        use_norm = kwargs.pop('use_norm', True)
        if use_norm:
            self._dv_norm = numpy.array([
                type(self).doc2vec(doc, self._embedding, use_norm=use_norm, **kwargs)
                for doc in self.parsed_documents
            ])
            self._dv = self._dv_norm
        else:
            self._dv = numpy.array([
                type(self).doc2vec(doc, self._embedding, use_norm=False, **kwargs)
                for doc in self.parsed_documents
            ])

    def writer2vec(self, **kwargs):
        """Pipeline for generating Author and document embeddings."""
        # NOTE: Ensure that parameter names do not collide.
        self.preprocess(kwargs.pop('tokenizer', Tokenizer(lemmatizer='wordnet')))
        self.partition_into_documents(
            size=kwargs.pop('part_size', None),
            remain_factor=kwargs.pop('remain_factor', 1.),
        )
        # Extract arguments for operations after embed(), because it consumes
        # remaining kwargs.
        stopwords = kwargs.pop('stopwords', None)
        func = kwargs.pop('func', np_avg)
        use_norm = kwargs.pop('use_norm', True)
        missing_value = kwargs.pop('missing_value', 0)
        self.embed(**kwargs)
        self.embed_documents(
            stopwords=stopwords,
            func=func,
            use_norm=use_norm,
            missing_value=missing_value,
        )

    def save(self, fn: str):
        """Save Author's state."""
        save_pickle(self, fn)

    @staticmethod
    def load(fn: str) -> 'Author':
        return load_pickle(fn)

    @staticmethod
    def doc2vec(
        doc: 'TextSpan',
        model: Union['gensim.models.word2vec', EmbeddingModel],
        *,
        stopwords: Iterable[str] = None,
        func: Callable = np_avg,
        use_norm: bool = True,
        missing_value: float = 0.,
    ) -> numpy.array:
        if stopwords is None:
            stopwords = set()

        if isinstance(model, EmbeddingModel):
            model = model.model

            missing_vector = numpy.empty(model.vector_size)
            missing_vector.fill(missing_value)
            vectors = []
            for word in doc.tokens:
                if word in stopwords or word not in model.wv:
                    vectors.append(missing_vector)
                else:
                    vectors.append(model.wv.word_vec(word, use_norm))
        return func(numpy.array(vectors))
