#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:38:23 2020
Synonyms Replacement Perturbation
Revising the output of Chris' synonym replacement;

1) For synonyms perturbation: 20% upper limit per document;
2) Synonym replacement using word vector similarity, part of speech, other model-agnostic qualities;

@author: Quan
"""

from nltk.corpus import wordnet
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import re
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
from preprocessing.tokenizer_rinehart import tokenizer

# Load a text file if required from ./data folder
path = './data/Doyle.txt'
data = open(path,"r+")
text = data.read()
# text = "Pete ate a large cake. Sam has a big mouth."


# Tokenize the text
tokenized = tokenizer(text)

# Get the list of words from the entire text
words = word_tokenize(text)

# Identify the parts of speech
tagged = nltk.pos_tag(words)

def replace(words, tagged):
    output = ""
    for i in range(0,len(words)):
        replacements = []

    # Only replace nouns with nouns, vowels with vowels etc.
        for syn in wordnet.synsets(words[i]):

            # Do not attempt to replace proper nouns or determiners
            if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                break

            # The tokenizer returns strings like NNP, VBP etc
            # but the wordnet synonyms has tags like .n.
            # So we extract the first character from NNP ie n
            # then we check if the dictionary word has a .n. or not
            word_type = tagged[i][1][0].lower()
            if syn.name().find("."+word_type+"."):
                # extract the word only
                r = syn.name()[0:syn.name().find(".")]
                replacements.append(r)

        if len(replacements) > 0:
            # Choose a random replacement
            replacement = replacements[randint(0,len(replacements)-1)]
            output = output + " " + replacement
        else:
            # If no replacement could be found, then just use the
            # original word
            output = output + " " + words[i]
    return output

text_out = replace(words, tagged)

# TODO: set 20% upper limit of synonyms perturbation per document
