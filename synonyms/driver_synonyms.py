#! /usr/bin/python3

import sys
from authordetect import save_text
from synonym_functions import perturb_author


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: {sys.argv[0]} tag rate infile outfile [embedding.bin]')
        sys.exit()

    tag = bool(int(sys.argv[1]))
    rate = float(sys.argv[2])
    infile, outfile = sys.argv[3:5]
    embedding_file = sys.argv[5] if len(sys.argv) == 6 else None
    print('Input file:', infile)
    print('Output file:', outfile)
    print('Embedding file:', embedding_file)

    perturbed_text, count = perturb_author(infile, embedding_file, tag, proportion=rate)

    save_text(perturbed_text, outfile)
