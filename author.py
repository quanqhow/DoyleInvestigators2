#! /usr/bin/python3

import re
import math
import functools
from smart_open import open
import facet
import sklearn
from typing import (
    Any,
    List,
    Tuple,
    Union,
    Iterable,
    Iterator,
    Callable,
)


# NOTE: The methods of this class are to be moved as functions to a file named textutils.py
class TextUtils:
    @staticmethod
    def merge_texts(srcs: Iterable[str], delim: str = '\n') -> str:
        """Get text content from multiple sources and join into a
        delimited string."""
        return delim.join(map(TextUtils.load_text, srcs))

    @staticmethod
    def get_text(src: str, *, phony: bool = False) -> str:
        """Get text content from a given source.

        Args:
            phony (bool): If set, 'src' is considered as the text content.
        """
        if not phony:
            try:
                with open(src) as fd:
                    return fd.read()
            except FileNotFoundError:
                pass
        return src

    @staticmethod
    def save_text(text: str, fn: str, *, flag: str = 'w') -> int:
        """Save a string to a file."""
        with open(fn, flag) as fd:
            return fd.write(text)

    @staticmethod
    def is_iter_not_str(obj: Any) -> bool:
        return hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes))

    @staticmethod
    def iter2str(
        data: Union[str, Iterable[Any]],
        delim: Union[str, Iterable[str]] = None,
    ) -> str:
        """Convert arbitrary iterable into a delimited string.

        Args:
            delim (str): Iterable of delimiters used to join strings at
                specific depths. Delimiter order is from top to lower levels.
                If empty delimiter, no delimiter is used throughout all levels.
                If only 1 delimiter, it is only used in top level.
                If multiple delimiters, it is used from top to lower levels.
        """
        if not TextUtils.is_iter_not_str(data):
            return data if isinstance(data, (str, bytes)) else str(data)
        if delim is None or len(delim) == 0:
            delim = ''
        return delim[0:1].join(map(lambda x: TextUtils.iter2str(x, delim[1:]), data))

        # elif len(delim) == 0:
        #     if TextUtils.is_iter_not_str(delim):
        #         delim = ''
        #     return delim.join(map(lambda x: TextUtils.iter2str(x, delim), data))
        # elif len(delim) == 1:
        #     if TextUtils.is_iter_not_str(delim):
        #         delim = delim[0]
        #     return delim.join(map(TextUtils.iter2str, data))
        # return delim[0].join(map(lambda x: TextUtils.iter2str(x, delim[1:]), data))


textutils = TextUtils()


def item_count(
    data: Iterable[Any],
    *,
    attr: str = None,
    func: Callable = None,
    **kwargs,
) -> int:
    """Count the number of elementary items in an arbitrary nested iterable.

    Args:
        attr (str): Object's attribute to use as the data for counting.
            This is used in nested structures that share a common attribute.

        func (callable): Transformation applied to data.

    Kwargs:
        Passed directly to attribute member (if callable) and/or to 'func'.
    """
    if attr is not None and hasattr(data, attr):
        _attr = getattr(data, attr)
        data = _attr(**kwargs) if callable(_attr) else _attr
    if func is not None:
        data = func(data, **kwargs)
    _item_count = functools.partial(item_count, attr=attr, func=func, **kwargs)
    return sum(map(_item_count, data)) if textutils.is_iter_not_str(data) else 1


def find_max_depth(
    data: Iterable[Any],
    *,
    dtype: Union[type, Tuple[type]] = (str, bytes),
) -> int:
    """Find maximum nested depth in iterable structures.

    Args:
        dtype (type): Object type to use as base case and stop recursion.
    """
    if isinstance(data, dtype) or len(data) == 0:
        return 0
    _depth = functools.partial(find_max_depth, dtype=dtype)
    return 1 + max(map(_depth, data))


def flatten(
    data: Iterable[Any],
    *,
    dtype: Union[type, Tuple[type]] = (str, bytes),
    df_mode: bool = True,
) -> Iterator[Any]:
    """Convert arbitrary iterable into a 1D iterator.

    Args:
        dtype (type): Object type to use as base case and stop recursion.

        df_mode (bool): If set, use depth-first traverse technique.
            If not set, use breadth-first instead.
    """
    if isinstance(data, dtype):
        yield data
    else:
        if df_mode:
            for item in data:
                yield from flatten(item, dtype=dtype)
        else:
            _data = []
            for item in data:
                if isinstance(item, dtype):
                    yield item
                else:
                    _data.append(item)
            yield from flatten(_data, dtype=dtype)





from typedlist import tlist
class TextSpan(tlist):
    """Composable lazy container representing a text span.

    Args:
        span ((int,int)): Index bounds from original string.
            End bound is non-inclusive to satisfy: text == orig[slice(*span)]

    Kwargs:
        Arguments passed directly to tlist().
    """
    def __init__(
        self,
        data: Union[str, Iterable['TextSpan']] = None,
        span: Iterable[int] = (0, 0),
        *,
        delim: str = ' ',
        **kwargs,
    ):
        # NOTE: Allow modifying these attributes to improve API use cases.
        self.__dict__['span'] = span
        self.__dict__['delim'] = delim
        kwargs['dtype'] = (str, bytes, type(self))
        super().__init__(
            [data] if isinstance(data, kwargs['dtype']) else data,
            **kwargs,
        )

    def __str__(self):
        # Generated string follows order of tokens, not their spans
        if len(self.delim) == 1:
            return textutils.iter2str(self, self.depth * self.delim)
        return textutils.iter2str(self, self.delim)

    @property
    def str(self):
        return str(self)

    @property
    def size(self):
        return sum(1 for _ in self.iter_tokens())

    @property
    def tokens(self):
        return list(self.iter_tokens())

    @property
    def vocabulary(self):
        return set(self.iter_tokens())

    @property
    def depth(self):
        return find_max_depth(self, dtype=str)

    def sort(self, **kwargs):
        """Recursive in-place sort. Defaults to sorting by spans."""
        kwargs['key'] = kwargs.get('key', lambda ts: ts.span)
        for item in self:
            if isinstance(item, type(self)):
                super().sort(**kwargs)
                item.sort(**kwargs)

    def count(self, value: str, *, exact_match: bool = False) -> int:
        """Search for value in text spans and return count."""
        return len(list(self.search(value, exact_match=exact_match)))

    # def count(self, value: str, *, exact_match: bool = False) -> int:
    #     if not exact_match:
    #        value = value.lower()
    #         pattern = re.compile(fr'\b{value}\b')
    #
    #     cnt = 0
    #     for item in self:
    #         if isinstance(item, type(self)):
    #             cnt += item.count(value)
    #         elif exact_match:
    #             cnt += int(item == value)
    #         else:
    #             cnt += len(pattern.findall(item.lower()))
    #     return cnt

    def search(
        self,
        value: str,
        *,
        exact_match: bool = False,
    ) -> Iterable[Iterable[int]]:
        """Search for value in text spans and return spans."""
        # Use regex to find tokens in strings or multi-word spans
        if not exact_match:
            value = value.lower()
            pattern = re.compile(fr'\b{value}\b')

        def search(obj, span=(0, 0)):
            for item in obj:
                if isinstance(item, type(self)):
                    yield from search(item, item.span)
                elif exact_match:
                    if item == value:
                        yield span
                else:
                    for _ in pattern.finditer(item.lower()):
                        yield span

        yield from search(self)

    def describe_span(self, span: Tuple[int, int]) -> Iterable[int]:
        """Find the hierarchy of structure indexes where a span is located."""
        def is_span_bounded_by(s1, s2) -> bool:
            """Checks if span1 is bounded by span2."""
            return s1[0] >= s2[0] and s1[1] <= s2[1]

        idxs = []
        # Early exit if span is empty
        if span[0] == span[1]:
            return idxs
        # for item in obj:
        #     if isinstance(item, type(self)):
        #         if is_span_bounded_by
        return idxs

    def iter_tokens(self, depth: int = -1):
        def iter_tokens(obj, curr_depth):
            for item in obj:
                if curr_depth < 0:
                    if isinstance(item, type(self)):
                        yield from iter_tokens(item, curr_depth)
                    else:
                        yield item
                elif curr_depth == depth:
                    if isinstance(item, type(self)):
                        yield str(item)
                    else:
                        yield item
                elif curr_depth < depth:
                    if isinstance(item, type(self)):
                        yield from iter_tokens(item, curr_depth + 1)
                    else:
                        yield item

        if depth < 0:
            yield from iter_tokens(self, depth)
        elif depth == 0:
            yield str(self)
        else:
            yield from iter_tokens(self, 1)

    # def iter_tokens(self):
    #     for item in self:
    #         if isinstance(item, type(self)):
    #             yield from item.iter_tokens()
    #         else:
    #             yield item


class Tokenizer:
    """Tokenizer.

        use_stopwords (bool): If set, use stopwords, else otherwise.

    Kwargs: (common options across facet.Tokenizers)
        converters (str, iterable[callable]): Available options are 'lower',
            'upper', 'unidecode'
    """
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
        self._sentencizer = facet.NLTKTokenizer(use_stopwords=use_stopwords, lemmatizer=lemmatizer, **kwargs)
        self._lemmatizer = None if lemmatizer is None else self._sentencizer._lemmatizer
        self._tokenizer = facet.WhitespaceTokenizer(use_stopwords=use_stopwords, min_token_length=min_token_length, **kwargs)

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
        if self._lemmatizer is not None:
            # Lemmatize selected parts-of-speech
            # NOTE: Do not lemmatize words not matching selected POS.
            for p in pos:
                text = self._lemmatizer.lemmatize(text, p)
        return text


class Author:
    def __init__(self, corpus: str = None):
        self._corpus = textutils.get_text(corpus) if corpus else corpus
        self._parsed = TextSpan()

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
            yield from self._parsed.iter_tokens(depth - 1)

    @property
    def sents(self):
        depth = self._parsed.depth
        if depth >= 3:
            yield from self._parsed.iter_tokens(depth - 2)

    def preprocess(
        self,
        tokenizer: 'Tokenizer' = None,
    ):
        if tokenizer is None:
            self._parsed = TextSpan(self._corpus, (0, len(self._corpus)))
            return

        sents = []
        for sb, se, s in tokenizer.sentencize(self._corpus):
            sent = []
            for tb, te, t in tokenizer.tokenize(s):
                _tspan = (tb + sb, te + sb)
                _t = tokenizer.lemmatize(t)
                sent.append(TextSpan(_t, _tspan))
            if len(sent) > 0:
                sents.append(TextSpan(sent, (sb, se)))
        self._parsed.extend(sents)

    def partition(self, size: int = 350, *, remain_factor: float = 1.):
        """Partition text into documents of a specified token count."""
        # Limit lower bound of size
        size = max(1, size)

        # Iterate through sentences
        cnt = 0
        doc = []
        docs = []
        for s in self.sents:
            cnt += item_count(s)
            if cnt <= size:
                # Add sentence to current document until partition size is satisfied
                doc.append(s)
                if spans:
                    doc_spans.append(spans[i])

                # Size limit not met, need more tokens
                if cnt < size:
                    continue
            else:
                # Truncate last sentence
                doc.append(s[:size - cnt])
                if spans:
                    doc_spans.append(spans[i])

            # Add current document to collection
            docs.append(doc)
            if spans:
                docs_spans.append(doc_spans)

            # Reset document controls
            cnt = 0
            doc = []
            doc_spans = []

        # Consider remaining string as a document if it is "long" enough
        if doc and item_count(doc) >= math.ceil(size * remain_factor):
            docs.append(doc)
            if spans:
                docs_spans.append(doc_spans)
        return (docs, docs_spans) if spans else docs


def tokenize2(text):
    import nltk
    import time

    wnl = nltk.stem.WordNetLemmatizer()

    # tokenize the data into sentences
    sent = nltk.tokenize.sent_tokenize(text)
    t = time.time()
    sent_ls = []
    for s in sent:
        # tokenize sentence into words
        s = re.split(' |\--', s)
        w_ls = []
        for w in s:
            w = w.lower()
            w = re.sub('[,"\.\'&\|:@>*;/=?!\']', "", w)
            w = re.sub('^[0-9\.]*$', "", w)
            w = re.sub("[^A-Za-z']+", " ", w)
            w = wnl.lemmatize(w, pos='v')
            w = wnl.lemmatize(w, pos='n')
            w_ls.append(w) # each list is a sentence
    # remove empty strings
    while "" in w_ls:
        w_ls.remove("")
    sent_ls.append(w_ls) # list of lists
    print('Time to clean up everything: {} mins'.format(round((time.time() - t) / 60, 2)))
    return sent_ls


def partition(
    text: Iterable[Iterable[str]],
    size: int = 350,
    *,
    remain_factor: float = 1.,
    spans: Iterable[Iterable[int]] = None,
) -> List[List[List[str]]]:
    """Partition text into documents of a specified token count."""
    # Limit lower bound of size
    size = max(1, size)

    # Iterate through sentences
    cnt = 0
    doc = []
    docs = []
    doc_spans = []
    docs_spans = []
    for i, s in enumerate(text):
        cnt += item_count(s)
        if cnt <= size:
            # Add sentence to current document until partition size is satisfied
            doc.append(s)
            if spans:
                doc_spans.append(spans[i])

            # Size limit not met, need more tokens
            if cnt < size:
                continue
        else:
            # Truncate last sentence
            doc.append(s[:size - cnt])
            if spans:
                doc_spans.append(spans[i])

        # Add current document to collection
        docs.append(doc)
        if spans:
            docs_spans.append(doc_spans)

        # Reset document controls
        cnt = 0
        doc = []
        doc_spans = []

    # Consider remaining string as a document if it is "long" enough
    if doc and item_count(doc) >= math.ceil(size * remain_factor):
        docs.append(doc)
        if spans:
            docs_spans.append(doc_spans)
    return (docs, docs_spans) if spans else docs


def shuffle_split(
    data: Iterable[Any],
    *,
    n_splits: int = 1,
    test_size: float = 0.1,
    **kwargs,
) -> Iterator[Iterable[Iterable[Any]]]:
    """Random permutation of iterable data.

    Returns: iter(train_idxs, test_idxs)
    """
    return sklearn.model_selection.ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        **kwargs,
    ).split(data)


def docs_split(
    docs: Iterable[Iterable[Iterable[str]]],
    *,
    sort: bool = False,
    **kwargs,
) -> Tuple[List[Iterable[Iterable[str]]]]:
    """Permute documents into training and test datasets."""
    train_idxs, test_idxs = next(shuffle_split(docs, **kwargs))

    # Sort indexes
    if sort:
        train_idxs.sort()
        test_idxs.sort()

    # Extract documents
    train = [docs[i] for i in train_idxs]
    test = [docs[i] for i in test_idxs]
    return train, test


def get_document_text_by_spans(text: str, spans: Iterable[Iterable[int]]) -> str:
    """Extract text from corpus corresponding to a single document."""
    return text[spans[0][0]:spans[-1][1]]


def get_documents_text_by_spans(
    text: str,
    spans: Iterable[Iterable[Iterable[int]]],
) -> str:
    """Extract text from corpus corresponding to multiple documents."""
    return list(map(lambda x: get_document_text_by_spans(text, x), spans))
