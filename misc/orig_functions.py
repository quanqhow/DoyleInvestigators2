#! /usr/bin/python3

import re
import math
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
)


def load_corpus(fn: str) -> str:
    """Read a string from a source."""
    with open(fn) as fd:
        return fd.read()


def save_corpus(text: str, fn: str, *, flag: str = 'w') -> int:
    """Save a string to a file."""
    with open(fn, flag) as fd:
        return fd.write(text)


def merge_files(fns: Iterable[str], delim: str ='\n') -> str:
    """Given a list of string sources, read and join into a delimited string."""
    return delim.join(map(load_corpus, fns))


def to_string(
    data: Union[str, Iterable[Any]],
    delim: Union[str, Iterable[str]] = ' ',
) -> str:
    """Convert arbitrary iterable into a delimited string.

    Args:
        delim: Iterable of delimiters used to join strings at specific depth.
            Delimiter order is from outer to innermost use.
    """
    if isinstance(data, str):
        return data
    elif len(delim) == 1:
        if not isinstance(delim, str):
            delim = delim[0]
        return delim.join(map(to_string, data))
    else:
        return delim[0].join(map(lambda x: to_string(x, delim[1:]), data))


def flatten(data: Iterable[Any], *, base_type: type = str) -> Iterator[Any]:
    """Convert arbitrary iterable into a 1D iterable."""
    if isinstance(data, base_type):
        yield data
    else:
        for x in data:
            for y in flatten(x):
                yield y


def token_count(text: Iterable[Any]) -> int:
    """Count the number of elementary tokens in an arbitrary nested list."""
    return (
        sum(map(token_count, text))
        if not isinstance(text, str)
        else 1
    )


def tokenize(
    text: str,
    delim: str = ' ',
    *,
    lemmatizer: str = 'wordnet',
    pos: Iterable[str] = 'vn',
    with_spans: bool = False,
) -> Tuple[List[List[str]], List[Tuple[int]]]:
    """Parse a string into a list of sentences each with a list of tokens."""

    # Create tokenizers for sentences, tokens, and lemmatize
    # Do not use stopwords
    tokenizer1 = facet.NLTKTokenizer(use_stopwords=False, lemmatizer=lemmatizer)
    lemmatizer = tokenizer1._lemmatizer
    tokenizer2 = facet.WhitespaceTokenizer(use_stopwords=False, min_token_length=1)

    # Regex for remove non-alpha and numeric tokens
    filter_pattern = re.compile(
        '('
        r"[^\w']"  # non-alphanumerics (except apostrophe to allow contractions)
        '|'
        r'\b\d+\b'  # standalone numbers
        ')',
        flags=re.ASCII,
    )

    # Sentencize
    sents = []
    spans = []
    for sb, se, s in tokenizer1.sentencize(text):
        sent = []
        # Apply filter and tokenize
        for _, _, t in tokenizer2.tokenize(filter_pattern.sub(delim, s)):
            if lemmatizer:
                # Lemmatize selected parts-of-speech
                # NOTE: Do not lemmatize words not matching selected POS.
                for p in pos:
                    t = lemmatizer.lemmatize(t, p)
            sent.append(t)
        else:
            sents.append(sent)
        if with_spans:
            spans.append((sb, se + 1))
    return (sents, spans) if with_spans else sents


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
        cnt += token_count(s)
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
    if doc and token_count(doc) >= math.ceil(size * remain_factor):
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