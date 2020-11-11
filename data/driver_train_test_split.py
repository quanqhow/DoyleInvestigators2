#! /usr/bin/python3

import os
import sys
from authordetect import Author, TextSpan, textutils, trainutils


def save_data_to_file(corpus: str, docs: 'TextSpan', outfile: str):
    # Create output directory
    # Use a temporary directory to prevent overwrites, user is responsible for
    # moving output to a valid destination.
    outdir = os.path.dirname(outfile)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Save data to file
    spans = [doc.span for doc in docs]
    textutils.save_text(' '.join(textutils.get_text_from_span(corpus, spans)), outfile)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: {sys.argv[0]} part_size infile trainfile testfile [test_size remain_factor seed]')
        print('part_size (int): number of words per partition')
        print('infile (str): Text file')
        print('trainfile (str): Text file')
        print('testfile (str): Text file')
        print('test_size (float): Test fraction')
        print('remain_factor (float): Partition remainder fraction')
        print('seed (int): Seed for split')
        sys.exit()

    part_size = int(sys.argv[1])
    infile, trainfile, testfile = sys.argv[2:5]
    test_size = float(sys.argv[5]) if len(sys.argv) > 5 else .1
    # Factor for capturing as much as possible from trailing text
    # Default is 1. but set to 0.1 because 3500/350=10%
    remain_factor = float(sys.argv[6]) if len(sys.argv) > 6 else 1.
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    print('Input file:', infile)
    print('Train file:', trainfile)
    print('Test file:', testfile)

    author = Author(infile)
    author.preprocess()
    author.partition_into_documents(part_size, remain_factor)
    print('Documents:', len(author.documents))
    print('Document tokens:', author.parsed_documents[0].size, author.parsed_documents[-1].size)

    # Train/test splits
    train_docs, test_docs = trainutils.split_data_into_train_test(
        author.parsed_documents,
        test_size=test_size,
        random_state=seed,
    )
    train_docs = TextSpan(train_docs)
    test_docs = TextSpan(test_docs)
    print('Training documents:', len(train_docs))
    print('Training tokens:', train_docs[0].size, train_docs[-1].size)
    print('Testing documents:', len(test_docs))
    print('Testing tokens:', test_docs[0].size, test_docs[-1].size)

    # Save train/test data to file
    save_data_to_file(author.text, train_docs, trainfile)
    save_data_to_file(author.text, test_docs, testfile)
