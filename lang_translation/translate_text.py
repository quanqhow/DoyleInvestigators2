#! /usr/bin/python3

import re
import json
from authordetect import Author


with open('translations.json') as fd:
    tables = json.load(fd)


def translate(text: str, to_country: str = 'uk', tag: bool = False):
    to_country = to_country.lower()
    if to_country not in ['us', 'uk']:
        raise ValueError("`country` must be one of ['us', 'uk']")

    count = 0
    conversion = 'us_to_uk' if to_country == 'uk' else 'uk_to_us'
    for source, target in tables[conversion].items():
        target = f'<{source}|{target}>' if tag else target
        text = text.replace(source, target)
        count += len(list(re.finditer(source, text)))
    return text, count


def get_documents(corpus_and_labels, part_size: int):
    if isinstance(corpus_and_labels, str):
        corpus_and_labels = [(corpus_and_labels, None)]
    docs = []
    for corpus, label in corpus_and_labels:
        author = Author(corpus, label)
        author.preprocess()
        author.partition_into_docs(part_size)
        for doc in author.docs:
            docs.append({
                'label': author.label,
                'text': str(doc),
            })
    return docs
