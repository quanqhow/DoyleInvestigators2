#! /usr/bin/python3

from authordetect import Author, EmbeddingModel
from tsne import (
    get_top_n_similar_words,
    get_top_n_disimilar_words,
    tsne_generation,
    plot_tsne,
)


#-----------------------Parameters------------------------
infile = '../data/Doyle_90.txt'
seed = 0
word = 'watson'
top = 14
n_components = 20


#-----------------------Embedding------------------------
# Method 1: Load existing embedding
embedding_file = '../training/doyle_50dim_350part.bin'
embedding = EmbeddingModel(embedding_file)

# Method 2: Compute embedding model
# embedding = None


#-----------------------Processing------------------------
# Load corpus
author = Author(infile)
print('Corpus characters:', len(author.text))

# Sentence segmentation and tokenization
author.preprocess()
print('Corpus sentences:', len(author.sentences))
print('Corpus tokens:', len(author.words))
print('Corpus vocabulary:', len(author.parsed_text.vocabulary))

# Create an author's word2vec embedding model
author.embed(embedding=embedding, seed=seed)
print('Embedding vocabulary:', len(author.embedding.vocabulary))
print('Embedding matrix:', author.embedding.vectors.shape)

embedding_size = embedding.vectors.shape[1]
w2v_model = author.embedding.model  # access Gensim's Word2Vec directly


#--------------Get similar and disimilar words------------
similar_words = get_top_n_similar_words(w2v_model, word, top)
disimilar_words = get_top_n_disimilar_words(w2v_model, word, top)

#---------------------------Get data from tSNE------------------------------
data, labels, colors = tsne_generation(w2v_model, word, n_components, similar_words, disimilar_words, embedding_size)
plot_tsne(data, word, labels, colors)
