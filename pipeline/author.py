#! /usr/bin/python3

import sys
sys.path.append('..')
from tokenizers import NLTKTokenizer, WhitespaceTokenizer
import re
import math
import textutils
import functools
from textspan import TextSpan
from typing import Any, List, Tuple, Union, Iterable


__all__ = ['Tokenizer', 'Author']


class Tokenizer:
    """Tokenizer.

        use_stopwords (bool): If set, use stopwords, else otherwise.

    Kwargs: (common options across tokenizers.*Tokenizer)
        converters (str, iterable[callable]): Available options are 'lower',
            'upper', 'unidecode'
    """
    STOPWORDS = NLTKTokenizer._STOPWORDS

    def __init__(
        self,
        *,
        use_stopwords=False,
        min_token_length=1,
        lemmatizer='wordnet',
        **kwargs,
    ):

        # Create tokenizers for sentences, tokens, and lemmatize
        # Do not use stopwords
        self._sentencizer = NLTKTokenizer(
            use_stopwords=use_stopwords,
            lemmatizer=lemmatizer,
            **kwargs,
        )
        self._lemmatizer = (
            None
            if lemmatizer is None
            else self._sentencizer._lemmatizer
        )
        self._tokenizer = WhitespaceTokenizer(
            use_stopwords=use_stopwords,
            min_token_length=min_token_length,
            **kwargs,
        )

        # Regexes to transform input text of tokenization process
        # NOTE: Transformations need to maintain same alignment so that
        # spans are consistent with input corpus.
        self._tokenizer_transforms = [
            # Remove non-alphanumerics (apostrophe for contractions)
            functools.partial(re.sub, r"[^\w']", ' ', flags=re.ASCII),
            # Remove enclosing quotes
            functools.partial(re.sub, r"'(\w+)'", r' \1 ', flags=re.ASCII),
        ]

        # Regexes to select valid tokens
        self._tokenizer_filters = [
            # Check that tokens include at least one alpha character
            functools.partial(re.search, r'[A-Za-z]', flags=re.ASCII),
        ]

    def sentencize(self, text: str, *, with_spans: bool = True):
        for b, e, t in self._sentencizer.sentencize(text):
            yield (b, e + 1, t) if with_spans else t

    def tokenize(self, text: str, *, with_spans: bool = True):
        # Apply transformations at the sentence level so that output
        # from tokenizer are individual tokens.
        for token_transform in self._tokenizer_transforms:
            text = token_transform(text)

        for b, e, t in self._tokenizer.tokenize(text):
            for token_filter in self._tokenizer_filters:
                if token_filter(t):
                    break
            else:
                continue
            yield (b, e + 1, t) if with_spans else t

    def lemmatize(self, text: str, pos: Iterable[str] = 'vn') -> str:
        """Lemmatize selected parts-of-speech."""
        if self._lemmatizer is not None:
            # NOTE: Do not lemmatize words not matching selected POS.
            for p in pos:
                text = self._lemmatizer.lemmatize(text, p)
        return text


class Author:
    def __init__(self, corpus: str = None):
        self._corpus = textutils.load_text(corpus) if corpus else corpus
        self._parsed = TextSpan()
        self._docs = TextSpan()
        self._embedding = None

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
    def embedding(self):
        return self._embedding

    def preprocess(self, tokenizer: 'Tokenizer' = None):
        # Reset because parsed corpus might have changed
        self._docs = TextSpan()

        if tokenizer is None:
            span = (0, len(self._corpus))
            self._parsed = TextSpan(self._corpus, span)
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
            if doc and doc.size >= math.ceil(size * remain_factor):
                doc.span = (doc[0].span[0], doc[-1].span[1])
                yield doc

        docs = list(partition(size, remain_factor))
        span = (docs[0].span[0], docs[-1].span[1])
        self._docs = TextSpan(docs, span)
