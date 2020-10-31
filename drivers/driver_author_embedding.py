#! /usr/bin/python3

from authordetect import Author, Tokenizer

# NOTE: Set PYTHONHASHSEED to constant value to have deterministic hashing
# across Python interpreter processes.
import os
os.environ['PYTHONHASHSEED'] = str(0)


######################
# User Configuration #
######################
infile = '../data/Doyle_10.txt'
workers = 1
seed = 0


##############
# Processing #
##############
# Load corpus
a = Author(infile)
print('Corpus characters:', len(a.corpus))

# Sentence segmentation and tokenization
a.preprocess(Tokenizer())
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))
print('Corpus vocabulary:', len(a.parsed.vocabulary))

# Create an author's word2vec embedding model
a.embed(workers=workers, seed=seed)
print('Embedding vocabulary:', len(a.model.vocabulary))
print('Embedding matrix:', a.model.vectors.shape)

# Access the embedding matrix
a.model.vectors


####################################
# Accessing Vectors and Vocabulary #
####################################
# Method 1
vocab = list(a.model.vocabulary)
print(vocab[:10])  # print 10 words from vocabulary
a.model['holmes']  # get vector associated with a word

# Method 2
w2v_model = a.model.model  # access Gensim's Word2Vec directly
# w2v_model.wv.vocab
vec = w2v_model.wv.word_vec('holmes')  # get vector associated with a word
print(vec)
