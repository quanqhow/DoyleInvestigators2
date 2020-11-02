#! /usr/bin/python3

import sys
from authordetect import load_text, save_text
from translate_text import translate


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} lang infile outfile [tag]')
        sys.exit()

    lang, infile, outfile = sys.argv[1:4]
    tag = sys.argv[4] if len(sys.argv) > 4 else False

    source_text = load_text(infile)
    translated_text = translate(source_text, lang, tag)

    save_text(translated_text, outfile)
