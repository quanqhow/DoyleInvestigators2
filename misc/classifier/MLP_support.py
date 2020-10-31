# fundamental operations that transform a document into model feature space

def tokenizer(text):
  return text.strip().split() # separate by whitespace

def parser(tokens):
  return [len(' '.join(tokens)), len(tokens)] # just count the characters and tokens, strings are not natively usable as features

