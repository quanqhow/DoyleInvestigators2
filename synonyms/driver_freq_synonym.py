#! /usr/bin/python3

import sys
from authordetect import Author, load_json, save_json
from synonym_functions import perturb_author


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} rate infile outfile [embedding.bin]')
        print('files are JSON files')
        sys.exit()

    rate = float(sys.argv[1])
    infile, outfile = sys.argv[2:4]
    embedding_file = sys.argv[4] if len(sys.argv) == 5 else None
    print('Input file:', infile)
    print('Output file:', outfile)
    print('Embedding file:', embedding_file)

    # Generate list of documents
    docs = load_json(infile)
    print('Total documents:', len(docs))

    total_word_count = 0
    total_repl_count = 0
    perturb_freq_map = {}
    for i, doc in enumerate(docs):
        perturbed_text, repl_count = perturb_author(doc['text'], embedding_file, proportion=rate)
        author = Author(perturbed_text)
        author.preprocess()
        perturb_freq_map[i] = repl_count / len(author.words)

        total_repl_count += repl_count
        total_word_count += len(author.words)

    print('Perturbation ratio:', total_repl_count / total_word_count)
    print('Total replacement count:', total_repl_count)
    print('Total word count:', total_word_count)
    save_json(perturb_freq_map, outfile)
