#! /usr/bin/python3

import sys
import json
import collections


def load_json(fn: str) -> list:
    with open(fn) as fd:
        return json.load(fd)


LABEL_MAP = {
    'doyle': 1,
    'christie': 0,
    'rinehart': 0,
}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} infile')
        print('infile (str): JSON file')
        sys.exit()

    infile = sys.argv[1]
    print('Input file:', infile)

    # Load documents from JSON file
    docs = load_json(infile)

    print('Loaded', len(docs), 'documents for testing')

    # Process documents
    # Each document is represented as a dictionary with a 'label' and 'text' field
    labels = collections.defaultdict(int)
    for i, doc in enumerate(docs):
        label = LABEL_MAP[doc['label'].lower()]
        labels[doc['label'].lower()] += 1
        # print('Processing document', i, 'with label', label, '...')
        # print(doc['text'])
    print('Label count:', labels)
