import json


def load_corpus(fn: str) -> list:
    with open(fn) as fd:
        return json.load(fd)


if __name__ == '__main__':
    test_file = 'doyle_to_rinehart_350_synonyms.json'
    doc_size = 350

    # Load documents from JSON file
    docs = load_corpus(test_file)
    num_docs = len(docs)

    print('Loaded', num_docs, 'documents for testing')

    # Process documents
    # Each document is represented as a dictionary with a 'label' and 'text' field
    for i, doc in enumerate(docs):
        print('Processing document', i, 'with label', doc['label'], '...')
        # print(doc['text'])
