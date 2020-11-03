#! /usr/bin/python3

from authordetect import Author, Tokenizer, EmbeddingModel
from tsne import (
    get_top_n_similar_words,
    get_top_n_disimilar_words,
    tsne_generation,
    plot_tsne,
)


#-----------------------Parameters------------------------
infile = '../data/Doyle_90.txt'
workers = 1
seed = 0
word = 'watson'
top = 14
n_components = 20


#-----------------------Embedding------------------------
# Method 1: Load existing embedding
embedding_file = '../serving/doyle_50dim_350part.bin'
embedding = EmbeddingModel()
embedding.load(embedding_file)

# Method 2: Compute embedding model
# embedding = None


#-----------------------Processing------------------------
# Load corpus
a = Author(infile)
print('Corpus characters:', len(a.corpus))

# Sentence segmentation and tokenization
a.preprocess(Tokenizer())
print('Corpus sentences:', len(a.sentences))
print('Corpus tokens:', len(a.words))
print('Corpus vocabulary:', len(a.parsed.vocabulary))

# Create an author's word2vec embedding model
a.embed(embedding=embedding, workers=workers, seed=seed)
print('Embedding vocabulary:', len(a.embedding.vocabulary))
print('Embedding matrix:', a.embedding.vectors.shape)

embedding_size = embedding.vectors.shape[1]
w2v_model = a.embedding.model  # access Gensim's Word2Vec directly


#--------------Get similar and disimilar words------------
similar_words = get_top_n_similar_words(w2v_model, word, top)
disimilar_words = get_top_n_disimilar_words(w2v_model, word, top)

#---------------------------Get data from tSNE------------------------------
data, labels, colors = tsne_generation(w2v_model, word, n_components, similar_words, disimilar_words, embedding_size)
plot_tsne(data, word, labels, colors)
