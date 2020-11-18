#! /usr/bin/python3

import sys
import json
from authordetect import Tokenizer, EmbeddingModel, load_json, load_pickle, np_avg, np_sum
from predict_classifier import predict
from writer2vec import writer2vec, flatten


LABEL_MAP = {
    'doyle': 1,
    'christie': 0,
    'rinehart': 0,
}


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: {sys.argv[0]} part_size infile embedding mlp')
        print('part_size (int)')
        print('infile (str): JSON test file')
        print('embedding (str): Embedding binary file')
        print('mlp (str): MLP pickle file')
        sys.exit()

    part_size = int(sys.argv[1])
    test_file, embedding_file, mlp_file = sys.argv[2:5]
    print('Test file:', test_file)
    print('Embedding file:', embedding_file)
    print('MLP file:', mlp_file)

    docs = load_json(test_file)

    writer2vec_params = {
        # Document partitioning
        'part_size': part_size,  # int, None=for standalong documents
        # 'remain_factor': 350/350,  # float [0,1], default=1

        # doc2vec
        'stopwords': Tokenizer.STOPWORDS,  # iterable[str], None
        'func': np_avg,  # callable
        'use_norm': True,  # bool
        'missing_value': 0,  # int
    }

    # Load embedding model
    embedding = EmbeddingModel(embedding_file)

    # Load classifier model
    mlp = load_pickle(mlp_file)

    for doc in docs:
        doc_data = doc['text']
        doc_label = LABEL_MAP[doc['label']]

        # Document vectors and labels
        vectors, labels = writer2vec(doc_data, doc_label, embedding=embedding, **writer2vec_params)

        # Flatten data
        test_vectors = flatten(vectors)
        test_labels = flatten(labels)

        # Predict
        predict_labels, classes, probabilities, metrics = predict(mlp, test_vectors, test_labels)

        print('Document label:', LABEL_MAP[doc['label']])
        print('\tTrue:', test_labels)
        print('\tPredict:', predict_labels)
        # print('Classes:', classes)
        # print('Probabilities:', probabilities)
        print('\t', metrics)
