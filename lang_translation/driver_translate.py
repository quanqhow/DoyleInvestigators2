#! /usr/bin/python3

import sys
from authordetect import load_text, save_text
from translate_text import translate


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} lang infile outfile')
        sys.exit()

    lang, infile, outfile = sys.argv[1:4]
    source_text = load_text(infile)
    translated_text = translate(source_text, lang)
    save_text(translated_text, outfile)
