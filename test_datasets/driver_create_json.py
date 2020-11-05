#! /usr/bin/python3

import sys
import random
from authordetect import Author, save_json


def save_corpus(data: list, fn: str):
    """Write a data structure into a JSON file."""
    with open(fn, 'w') as fd:
        json.dump(data, fd)


def get_documents(corpus_and_labels, part_size: int = None):
    if isinstance(corpus_and_labels, str):
        corpus_and_labels = [(corpus_and_labels, None)]
    docs = []
    for corpus, label in corpus_and_labels:
        author = Author(corpus, label)
        author.preprocess()
        author.partition_into_docs(part_size)
        for doc in author.docs:
            words = Author.get_tokens(doc)
            docs.append({
                'label': author.label,
                'text': Author.substitute(author.corpus, words),
            })
    return docs


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: {sys.argv[0]} part_size outfile infile label [...infile label]')
        print('part_size (int): number of words per partition')
        print('outfile (str): JSON file')
        print('infile (str): text file')
        print('label (str): a word')
        sys.exit()

    if len(sys.argv) % 2 == 0:
        raise Exception('invalid number of arguments')

    part_size = int(sys.argv[1])
    outfile = sys.argv[2]

    infiles_and_labels = []
    for i in range(3, len(sys.argv), 2):
        infile, label = sys.argv[i:i+2]
        infiles_and_labels.append((infile, label))
    print('Input files/labels:', infiles_and_labels)
    print('Output file:', outfile)

    # Generate list of documents
    if part_size > 0:
        docs = get_documents(infiles_and_labels, part_size)
    else:
        docs = get_documents(infiles_and_labels)
    print('Total documents:', len(docs))

    # Randomize documents
    random.shuffle(docs)

    # Save documents to JSON file
    save_json(docs, outfile)
