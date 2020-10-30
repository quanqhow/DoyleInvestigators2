import json
import nltk
import random


def save_corpus(data: list, fn: str):
    """Write a data structure into a JSON file."""
    with open(fn, 'w') as fd:
        json.dump(data, fd)


def generate_dummy_doc(size):
    """Create a dummy document."""
    seed_text = 'The quick brown fox jumps over the lazy dog.'
    tokens = nltk.word_tokenize(seed_text)
    text = nltk.Text(tokens)
    return text.generate(size)


if __name__ == '__main__':
    test_file = 'doyle_to_rinehart_350_synonyms.json'
    doc_size = 350
    num_docs = 5
    labels = ['rinehart', 'doyle', 'christie']

    # Generate list of documents
    # Each document is represented as a dictionary with a 'label' and 'text' field
    docs = []
    for i in range(num_docs):
        doc = {
            # NOTE: In the real case, document labels are not randomized but set appropiately
            'label': labels[random.randint(0, len(labels) - 1)],
            'text': generate_dummy_doc(doc_size),
        }
        docs.append(doc)

    # Save documents to JSON file
    save_corpus(docs, test_file)
