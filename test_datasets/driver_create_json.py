#! /usr/bin/python3

import sys
import random
from authordetect import Author, Tokenizer, save_json


def get_documents(corpus_and_labels, part_size: int = None):
    if isinstance(corpus_and_labels, str):
        corpus_and_labels = [(corpus_and_labels, None)]
    docs = []
    for corpus, label in corpus_and_labels:
        author = Author(corpus, label)
        author.preprocess(Tokenizer(lemmatizer='wordnet'))
        author.partition_into_documents(part_size)
        for doc in author.parsed_documents:
            words = doc.get_tokens()
            docs.append({
                'label': author.label,
                'text': words.substitute(author.text),
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
