#! /usr/bin/python3

import sys
from authordetect import Author, load_json, save_json
from translate_text import translate


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} lang infile outfile')
        print('lang (str): uk, us')
        print('infile (str): JSON file')
        print('outfile (str): JSON file')
        sys.exit()

    lang, infile, outfile = sys.argv[1:]
    print('Input file:', infile)
    print('Output file:', outfile)

    # Generate list of documents
    docs = load_json(infile)
    print('Total documents:', len(docs))

    total_word_count = 0
    total_repl_count = 0
    perturb_freq_map = {}
    for i, doc in enumerate(docs):
        perturbed_text, repl_count = translate(doc['text'], lang)
        author = Author(perturbed_text)
        author.preprocess()
        perturb_freq_map[i] = repl_count / len(author.words)

        total_repl_count += repl_count
        total_word_count += len(author.words)

    print('Perturbation ratio:', total_repl_count / total_word_count)
    print('Total replacement count:', total_repl_count)
    print('Total word count:', total_word_count)
    save_json(perturb_freq_map, outfile)
