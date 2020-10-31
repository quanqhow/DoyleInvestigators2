#! /usr/bin/python3

import math
from .textutils import load_text
from .textspan import TextSpan


__all__ = ['Author']


class Author:
    def __init__(self, corpus: str = None):
        self._corpus = load_text(corpus) if corpus else corpus
        # Print info on how is input considered so that if a filename does
        # not exists, then user can be informed.
        if self._corpus == corpus:
            print('Input Mode: Author was provided raw text')
        else:
            print('Input Mode: Author was provided text file')
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
            if doc and doc.size >= math.ceil(size * remain_factor):
                doc.span = (doc[0].span[0], doc[-1].span[1])
                yield doc

        docs = TextSpan(list(partition(size, remain_factor)))
        if len(docs) > 0:
            docs.span = (docs[0].span[0], docs[-1].span[1])
        self._docs = docs
