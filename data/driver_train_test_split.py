#! /usr/bin/python3

import sys
from authordetect import Author, TextSpan, textutils, trainutils


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} part_size infile [trainfile testfile test_size remain_factor seed]')
        print('part_size (int): number of words per partition')
        print('infile (str): Text file')
        print('trainfile (str): Text file')
        print('testfile (str): Text file')
        print('test_size (float): Test fraction')
        print('remain_factor (float): Partition remainder fraction')
        print('seed (int): Seed for split')
        sys.exit()

    part_size = int(sys.argv[1])
    infile = sys.argv[2]
    trainfile = sys.argv[3] if len(sys.argv) > 3 else None
    testfile = sys.argv[4] if len(sys.argv) > 4 else None
    test_size = float(sys.argv[5]) if len(sys.argv) > 5 else .1
    # Factor for capturing as much as possible from trailing text
    # Default is 1. but set to 0.1 because 3500/350=10%
    remain_factor = float(sys.argv[6]) if len(sys.argv) > 6 else (350 / part_size)
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
    if trainfile:
        spans = [doc.span for doc in train_docs]
        text = ' '.join(textutils.get_text_from_span(author.text, spans))
        textutils.save_text(text, trainfile)

    if testfile:
        spans = [doc.span for doc in test_docs]
        text = ' '.join(textutils.get_text_from_span(author.text, spans))
        textutils.save_text(text, testfile)
