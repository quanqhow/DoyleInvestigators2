from MLP_support import tokenizer,parser

import sys
import pickle
from sklearn.neural_network import MLPClassifier

args = sys.argv
if len(args) < 3:
  print("Usage; python",args[0],"<saved model.pkl> <input text.txt> <OPTIONAL: output labels.txt>")
  exit(-1)

clf = pickle.load(open(args[1],"rb"))

# create input to be classified;
input_features = list()
infile = open(args[2],"r")
for line in infile:
  parsed = parser(tokenizer(line))
  input_features.append(parsed)
infile.close()

predictions = clf.predict(input_features)
if len(args) == 4:
  outfile = open(args[3],"w")
  for element in predictions:
    outfile.write(str(element) + "\n")
  outfile.close()
else:
  print(predictions)
