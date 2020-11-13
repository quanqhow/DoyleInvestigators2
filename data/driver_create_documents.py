#! /usr/bin/python3

import sys
from authordetect import Author, TextSpan, textutils, save_json


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} part_size infile [outfile]')
        print('part_size (int): number of words per partition')
        print('infile (str): Text file')
        print('outfile (str): JSON file')
        print('remain_factor (float): Partition remainder fraction')
        sys.exit()

    part_size = int(sys.argv[1])
    infile = sys.argv[2]
    outfile = sys.argv[3] if len(sys.argv) > 3 else None
    print('Input file:', infile)
    print('Output file:', outfile)

    author = Author(infile)
    author.preprocess()
    author.partition_into_documents(part_size)
    print('Documents:', len(author.documents))
    print('Document tokens:', author.parsed_documents[0].size, author.parsed_documents[-1].size)

    if outfile:
        text = [
            textutils.get_text_from_span(author.text, doc.span)
            for doc in author.parsed_documents
        ]
        save_json(text, outfile)
