import nltk
from nltk.stem import WordNetLemmatizer
import re
import time
wnl = WordNetLemmatizer()

path = '../data/Rinehart.txt'
data = open(path,"r+")
data = data.read()

def tokenizer(data):
  # tokenize the data into sentences
  sent = nltk.tokenize.sent_tokenize(data)
  t = time.time()
  sent_ls = []
  for s in sent:
    # tokenize sentence into words
    s = re.split(' |\--', s)
    w_ls = []
    for w in s:
      w = w.lower()
      w = re.sub('[,"\.\'&\|:@>*;/=?!\']', "", w)
      w = re.sub('^[0-9\.]*$', "", w)
      w = re.sub("[^A-Za-z']+", " ", w)
      w = wnl.lemmatize(w, pos='v')
      w = wnl.lemmatize(w, pos='n')
      w_ls.append(w) # each list is a sentence
    # remove empty strings
    while "" in w_ls:
      w_ls.remove("")
    sent_ls.append(w_ls) # list of lists
  print('Time to clean up everything: {} mins'.format(round((time.time() - t) / 60, 2)))
  return sent_ls
