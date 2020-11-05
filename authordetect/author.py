#! /usr/bin/python3

import numpy
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
    def __init__(self, corpus: str, label: Any = None):
        self._corpus = load_text(corpus) if corpus else corpus
        # Print info on how is input considered so that if a filename does
        # not exists, then user can be informed.
        if self._corpus is not None:
            if self._corpus == corpus:
                print('Author corpus was provided as raw text')
            else:
                print('Author corpus will be loaded from a file')
        self._label = label
        self._parsed = TextSpan()
        self._docs = TextSpan()
        self._embedding = None  # EmbeddingModel
        self._docs_vectors = numpy.array([])
        self._docs_vectors_norm = numpy.array([])

    @property
    def corpus(self):
        return self._corpus

    @property
    def label(self):
        return self._label

    @property
    def parsed(self):
        return self._parsed

    @property
    def words(self):
        return type(self).get_tokens(self._parsed, min_depth=2)

    @property
    def sentences(self):
        return type(self).get_tokens(self._parsed, min_depth=3)

    @property
    def docs(self):
        return self._docs

    # NOTE: *_str properties are utility methods useful to get lists of strings.
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
    def embedding(self):
        return self._embedding

    @property
    def docs_vectors(self):
        return self._docs_vectors

    @property
    def docs_vectors_norm(self):
        return self._docs_vectors_norm

    @staticmethod
    def get_tokens(text_span: 'TextSpan', *, min_depth=2):
        depth = text_span.depth
        if depth >= min_depth:
            tokens = list(text_span.iter_tokens(depth - (min_depth - 1)))
            return TextSpan(tokens, (tokens[0].span[0], tokens[-1].span[1]))
        return TextSpan()

    @staticmethod
    def substitute(
        text: str,
        text_span: 'TextSpan',
        *,
        force_capitalization: bool = True,
    ):
        """Substitutes the given tokens in a text."""
        parsed_text = ''
        for i, tspan in enumerate(text_span):
            # Get text before first new token, if token is "at the beginning"
            if i == 0 and tspan.span[0] < 3:
                parsed_text = text[:tspan.span[0]]

            # Check capitalization for new token.
            # First letter capitalization is applied if any of the first two
            # characters of original token are uppercase to allow a symbol
            # (e.g., quotation mark) and letter combination.
            # NOTE: This is tricky in the sense that if a substituted token is
            # not wanted capitalized, this will force it. Also, this is
            # tokenizer dependent.
            new_token = str(tspan)
            old_token = text[tspan.span[0]:tspan.span[1]]
            if force_capitalization and any(map(str.isupper, old_token[:2])):
                if old_token.isupper():
                    # Caplitalize token
                    new_token = new_token.upper()
                elif len(new_token) > 0:
                    # Capitalize only first character
                    new_token = new_token[0].upper() + new_token[1:]

            # Get text from end of new token to begin of next token or end
            # of text
            post_text = ''
            if i < len(text_span) - 1:
                post_text = text[slice(tspan.span[1], text_span[i+1].span[0])]

            # Substitute new token
            parsed_text += new_token + post_text
        return parsed_text

    def preprocess(self, tokenizer: Tokenizer = Tokenizer()):
        # Reset because parsed corpus might have changed
        self._docs = TextSpan()
        self._embedding = None
        self._docs_vectors = numpy.array([])
        self._docs_vectors_norm = numpy.array([])

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

    def partition_into_docs(self, size: int = None, remain_factor: float = 1.):
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

        # If no partition size provided, then consider a single document
        if size is None or size < 1:
            docs = TextSpan([self.sentences])
        else:
            docs = TextSpan(list(partition(size, remain_factor)))

        if len(docs) > 0:
            docs.span = (docs[0].span[0], docs[-1].span[1])
        self._docs = docs

    def embed(self, embedding=None, **kwargs):
        # Reset document embeddings
        self._docs_vectors = numpy.array([])
        self._docs_vectors_norm = numpy.array([])

        if embedding is None:
            self._embedding = EmbeddingModel(**kwargs)
            self._embedding.train(self.sentences_words_str)
        else:
            self._embedding = embedding

    def embed_docs(self, embedding=None, **kwargs):
        # NOTE: Auto-embed with default parameters
        if self._embedding is None or embedding is not None:
            self.embed(embedding)

        # Use norm vectors
        use_norm = kwargs.pop('use_norm', True)
        if use_norm:
            self._docs_vectors_norm = numpy.array([
                type(self).doc2vec(doc, self._embedding, use_norm=use_norm, **kwargs)
                for doc in self.docs
            ])
            self._docs_vectors = self._docs_vectors_norm
        else:
            self._docs_vectors = numpy.array([
                type(self).doc2vec(doc, self._embedding, use_norm=False, **kwargs)
                for doc in self.docs
            ])

    def writer2vec(self, **kwargs):
        """Pipeline for generating Author and document embeddings."""
        # NOTE: Ensure that parameter names do not collide.
        self.preprocess(kwargs.pop('tokenizer', Tokenizer()))
        self.partition_into_docs(
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
        self.embed_docs(
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
