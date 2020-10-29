import random
import numpy as np
from typing import List, Callable
from nltk.corpus import wordnet
import nltk

# Take two numpy vectors of the same length and return their cosine similarity.
def cosine_similarity(vec1,vec2):
  if len(vec1) != len(vec2):
    print("Cannot compute cosine similarity between vectors of mismatched lengths",len(vec1),len(vec2))
    return -2
  else:
    dotprod = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
      print("Cosine similarity: Cannot compare vectors with 0 length.")
      return 0
    cossim = dotprod / (norm1 * norm2)
    return cossim

def synonyms_NLTK(word : str) -> List[str]:
  """
  Select synonyms for a given word using NLTK wordnet
  :param word: Word for which to find synonyms
  :return: List of synonyms for given word
  """  
  return list(set([lemma.name() for synset in wordnet.synsets(word) for lemma in synset.lemmas()]))

def select_all(tokens):
  return [True] * len(tokens)

def select_random(tokens, proportion=0.5):
  return_vals = [False] * len(tokens)
  selections = random.choices(range(len(tokens)), k=proportion*len(tokens))
  for selection in selections:
    return_vals[selection] = True
  return return_vals

def replace_synonym(word, pos, synonym_function = synonyms_NLTK, embedding=None, cosine_threshold = 0.9):
  # generate synonym list
  synonyms = synonym_function(word)
  # cull by pos
  parts_of_speech = [nltk.pos_tag([synonym],tagset='universal')[0][1] for synonym in synonyms]
  tokens = [token for [token,token_pos] in zip(synonyms,parts_of_speech) if token_pos == pos] 
  # cull by embedding, if provided
  if embedding is not None and word in embedding: # can't check similarity if the word we're replacing isn't known
    word_vec = embedding[word]
    remaining_tokens = [token for token in tokens if token in embedding and cosine_similarity(word_vec, embedding[token]) > cosine_threshold]
  else:
    remaining_tokens = tokens
  # if nothing's left, return the original word
  if len(remaining_tokens) == 0:
    return word
  # otherwise, randomly select from what remains
  else:
    return random.choice(remaining_tokens)

def perturb_document(tokens, selection_function=select_all, replacement_function=replace_synonym):
  # get part of speech
  # this code unzips the pos output from [[word,pos],[word,pos]] to [[word,word],[pos,pos]] and then just takes the pos
  parts_of_speech = list(zip(* nltk.pos_tag(tokens,tagset='universal')))[1]

  selection = selection_function(tokens)

  for index, choice in enumerate(selection):
    if choice:
      word = tokens[index]
      pos = parts_of_speech[index]
      replacement = replacement_function(word,pos)
      tokens[index] = replacement

  return tokens
