import sys
from sklearn.neural_network import MLPClassifier
import pickle
from MLP_support import tokenizer,parser

args = sys.argv
if len(args) < 4:
  print("Usage;",args[0],"<input true.txt> <input false.txt> <output model.pkl>")
  exit(-1)

# intake documents and transform them into features
in_doc_features = list()
in_labels = list()
infile = open(args[1],"r")
for line in infile:
  parsed = parser(tokenizer(line))
  in_doc_features.append(parsed)
  in_labels.append(1)
infile.close()
infile = open(args[2],"r")
for line in infile:
  parsed = parser(tokenizer(line))
  in_doc_features.append(parsed)
  in_labels.append(0)
infile.close()

print(in_doc_features)
print(in_labels)

# construct and train classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clf.fit(in_doc_features, in_labels)


# example prediction;
print("Example prediction of doc 0 in `true` dataset;")
prediction = clf.predict([in_doc_features[0]])
print(prediction)

# save model for later usage
print("Pickling model in",args[3])
outfile = open(args[3],"wb")
pickle.dump(clf,outfile)
outfile.close()
