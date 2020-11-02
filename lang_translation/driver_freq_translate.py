#! /usr/bin/python3

import sys
from authordetect import Author, load_json, save_json
from translate_text import translate, get_documents


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} lang infile outfile')
        print('files are JSON files')
        sys.exit()

    lang, infile, outfile = sys.argv[1:]
    print('Input file:', infile)
    print('Output file:', outfile)

    # Generate list of documents
    docs = load_json(infile)
    print('Total documents:', len(docs))

    perturb_freq_map = {}
    for i, doc in enumerate(docs):
        translated_text, repl_count = translate(doc['text'], lang)
        author = Author(translated_text)
        author.preprocess()
        perturb_freq_map[i] = repl_count / len(author.words)

    save_json(perturb_freq_map, outfile)
