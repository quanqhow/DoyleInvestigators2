#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sns.set_style("darkgrid")


def get_top_n_similar_words(model, word, top_n):
    """
    This method return the n most frequent words related to a specific word.

    Parameters
        model:  (Word2Vec) model that was trained over a corpus.
        word:   (string) look for similar words to this one.
        top_n:  (int) indicates the number of words in ascending order.
    -
    Returns
        close_words: (tuple <string, float>)
                     string: word string
                     float: similarity index

    """
    close_words = model.wv.most_similar([word], topn=top_n)
    return close_words


def get_top_n_disimilar_words(model, word, negative_top):
    """
    This method return the n most disimilar words related to a specific word.

    -
    Parameters
        model:  (Word2Vec) model that was trained over a corpus.
        word:   (string) look for similar words to this one.
        negative_top:  (int) indicates the number of words in descending order.
    -
    Returns
        close_words: (tuple <string, float>)
                     string: word string
                     float: similarity index

    """
    close_words = model.wv.most_similar(negative=[word], topn=negative_top)
    return close_words


def tsne_generation(model, word, n_components, similar_words, disimilar_words, embedding_size):
    """
    Get the data from tSNE.

    -Parameters
        model:  (Word2Vec) model that was trained over a corpus.
        word:   (string) word to be analized.
        n_components: (int) number of components to apply PCA.
        similar_words: (list) list of top similar words precalculated.
        disimilar_words: (list) list of top disimilar words precalculated.
        embedding_size: (int) embedding size of the words.

    -Returns
        new_data: (list) list of dim reduced data obtained from PCA and then, tSNE.
        word_labels: (list) strings of the words (we need this to plot the results).
        color_list: (list) colors of the list just to plotting the results.
    """

    #1. Prepare the variables
    #---------------------------------------------------------------------------
    word_embeddings = np.empty((0, embedding_size), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds word embedding to search into the results array
    word_embeddings = np.append(word_embeddings, model.wv.__getitem__([word]), axis=0)
    #---------------------------------------------------------------------------


    #2. Get the closest words embeddings
    #---------------------------------------------------------------------------
    for wrd_score in similar_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        word_embeddings = np.append(word_embeddings, wrd_vector, axis = 0)
    #---------------------------------------------------------------------------


    #3. Adds the vector for each of the words from list_names to the array of embeddings
    #---------------------------------------------------------------------------
    for wrd_tuple in disimilar_words:
        wrd = wrd_tuple[0]
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        word_embeddings = np.append(word_embeddings, wrd_vector, axis = 0)
    #---------------------------------------------------------------------------


    # 4. Reduces the dimensionality of the embeddings to the number <<n_components>> dimensions with PCA
    #---------------------------------------------------------------------------
    reduc = PCA(n_components=n_components).fit_transform(word_embeddings)
    #---------------------------------------------------------------------------


    # 5. Finds t-SNE coordinates for 2 dimensions
    #---------------------------------------------------------------------------
    np.set_printoptions(suppress=True)
    new_data = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    #---------------------------------------------------------------------------

    return new_data, word_labels, color_list


def plot_tsne(data, main_word, word_labels, color_list):
    """
    Plot the results from the tSNE method.

    -
    Parameters:
        data: (list) tSNE data.
        main_word: word from which we want to plot.
        word_labels: (list) strings of the words (we need this to plot the results).
        color_list: (list) colors of the list just to plotting the results.

    -
    Returns:
        None.
    """

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in data[:, 0]],
                       'y': [y for y in data[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    plt.xlim(data[:, 0].min()-50, data[:, 0].max()+50)
    plt.ylim(data[:, 1].min()-50, data[:, 1].max()+50)
    plt.title('t-SNE visualization for {}'.format(main_word.title()))#'''
    plt.show()
