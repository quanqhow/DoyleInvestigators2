#! /usr/bin/python3

import json


def load_corpus(fn: str) -> list:
    with open(fn) as fd:
        return json.load(fd)


LABEL_MAP = {
    'doyle': 1,
    'christie': 0,
    'rinehart': 0,
}


if __name__ == '__main__':
    test_file = 'perturbed_langtranslation_rinehart_350.json'

    # Load documents from JSON file
    docs = load_corpus(test_file)
    num_docs = len(docs)

    print('Loaded', num_docs, 'documents for testing')

    # Process documents
    # Each document is represented as a dictionary with a 'label' and 'text' field
    for i, doc in enumerate(docs):
        label = LABEL_MAP[doc['label'].lower()]
        print('Processing document', i, 'with label', label, '...')
        # print(doc['text'])
