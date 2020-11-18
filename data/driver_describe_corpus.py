#! /usr/bin/python3

import sys
import random
from authordetect import Author, load_json


def describe_corpus(corpus: dict):
    total_sents = 0
    total_words = 0
    total_chars = 0
    for idx, (story, text) in enumerate(corpus.items(), start=1):
        author = Author(text)
        author.preprocess()
        print(f'{idx}.', story)
        print('\tSentences:', len(author.sentences))
        print('\tWords:', len(author.words))
        print('\tCharacters:', len(author.text))
        total_sents += len(author.sentences)
        total_words += len(author.words)
        total_chars += len(author.text)
    print('Total sentences:', total_sents)
    print('Total words:', total_words)
    print('Total characters:', total_chars)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} infile')
        print('infile (str): JSON file')
        sys.exit()

    infile = sys.argv[1]
    print('Input file:', infile)

    # Load documents from JSON file
    corpus = load_json(infile)
    print('Loaded', len(corpus), 'texts for describing')

    describe_corpus(corpus)
