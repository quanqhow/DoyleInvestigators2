#! /usr/bin/python3

import sys
from authordetect import load_text, save_text
from translate_text import translate


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} lang infile outfile [tag]')
        print('lang (str): uk, us')
        print('infile (str): text file')
        print('outfile (str): text file')
        print('tag (int): 0 = False, 1 = True')
        sys.exit()

    lang, infile, outfile = sys.argv[1:4]
    tag = sys.argv[4] if len(sys.argv) > 4 else False
    print('Input file:', infile)
    print('Output file:', outfile)

    source_text = load_text(infile)
    translated_text, _ = translate(source_text, lang, tag)

    save_text(translated_text, outfile)
