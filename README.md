# Doyle Investigators - Vector Embeddings and Adversarial Analysis

This project constructs a binary classifier for Sir Arthur Conan Doyle using
a dataset of Sherlock Holmes novels and short stories.


## General Information

The `authordetect` package follows a modular object-oriented approach.
The most relevant classes are:
* `Author` (*authordetect/author.py*) - This class represents a corpus
  corresponding to a single author and provides capabilities to load and
  tokenize corpus, partition into documents, create embedding models for author
  and each document. All these actions are part of the `writer2vec` algorithm
  (see Overleaf paper), and a method with the same name is provided that
  applies these transformations as a single step.
* `Tokenizer` (*authordetect/tokenizer.py*) - This class represents a tokenizer
  for performing sentence segmentation and tokenization of an `Author's`
  corpus. It also contains a list of stopwords (from NLTK).
* EmbeddingModel (*authordetect/embedding.py*) - This class represents a
  vector embedding model and is a wrapper over Gensim's Word2Vec with added
  capabilities to save/load embeddings and ease of use. Embedding with normalized
  vectors are used by default.
* `Classifier` (*authordetect/classifier*) - This class represents a MLP
  classifier and is used to train on document vectors (with corresponding
  lables). Afterwards, it can provide predictions on new document vectors.

For reproducible results, set the `seed` paramater during training and prediction.
Also, set environment variable `PYTHONHASHSEED` to an integer prior to launching
Python interpreter process.


## Installation

* The following packages are required (see `requirements.txt`):
  * Python 3.6 or greater
  * typing
  * configparser
  * unidecode
  * urllib3
  * smart\_open
  * bs4
  * psutil
  * nltk
  * gensim
  * scikit-learn
  * pandas
  * seaborn
  * matplotlib
  * numpy


### Local Install

* Install package and dependencies on a local system
  ```shell
  > git clone https://github.com/edponce/DoyleInvestigators2.git
  ```

* Create a virtual environment (Anaconda)
  ```shell
  > conda create -n authordetect python=3.7
  > cd DoyleInvestigators2
  > pip install -e .
  > python setup_nltk.py
  > python
  ```

* See `Usage` section below.
  ```python
  >>> import authordetect
  >>> ...
  ```


### Google Colab Install
* See example notebook in `drivers/AuthorDetect_AuthorEmbedding.ipynb`.
  The code is download directly from GitHub repo and installed.
  ```python
  >>> !pip install git+https://github.com/edponce/DoyleInvestigators2
  >>> # May need to restart runtime so that correct package versions are loaded
  ```
  Set up NLTK:
  ```python
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> nltk.download('punkt')  # sentencizer
  >>> nltk.download('averaged_perceptron_tagger')  # tagger
  >>> nltk.download('universal_tagset')  # universal POS tags
  >>> nltk.download('wordnet')  # lemmatizer
  ```
  For data files, you need to mount the Google Drive so that the folder shared
  with corpus data is visible for notebook.
  ```python
  >>> from google.colab import drive
  >>> drive.mount('/content/gdrive')
  ```
  Now you should be able to run `authordetect`:
  ```python
  >>> from authordetect import Author
  >>> infile = '/content/gdrive/My Drive/.../text.txt'
  >>> author = Author(infile)
  >>> ...
  ```


## Usage

### Example: Create an author's embedding matrix

  ```python
  >>> # Load an author's corpus
  >>> from author import Author, Tokenizer
  >>> author = Author('data/Doyle_10.txt')
  >>> author.corpus  # this is the raw text
  >>>
  >>> # Preprocess text without removing stopwords
  >>> tokenizer = Tokenizer(use_stopwords=False)
  >>> author.preprocess(tokenizer)
  >>>
  >>> # Create an author's word2vec embedding model
  >>> author.embed()
  >>> author.embedding.vocabulary  # access vocabulary from entire corpus
  >>> author.embedding.vectors  # access non-normalized embedding matrix (NumPy 2D array)
  >>> author.embedding.vectors_norm  # access normalized embedding matrix (NumPy 2D array)
  >>> author.embedding['holmes']  # get vector associated with a word
  ```

### Example: Save and load author's embedding model

* Save Gensim's Word2Vec model:
  ```python
  >>> author.embedding.save('my_embedding.bin')
  ```

* Load existing Gensim's Word2Vec model:
  ```python
  >>> from authordetect import Author, EmbeddingModel
  >>> embedding = EmbeddingModel()
  >>> embedding.load('my_embedding.bin')
  >>>
  >>> # Use the loaded embedding with an Author
  >>> author = Author('text.txt')
  >>> author.preprocess()
  >>> author.embed(embedding)
  ```


## Datasets and Models

* MLP classifier models and author embeddings were created with the
  `serving/driver_train.py` script setting `seed=0`, `PYTHONHASHSEED=0`, and
  `remain_factor=350/<part_size>`.

* US/UK English translation was performed to entire corpus.
  ```shell
  > cd lang_translation/
  > python driver_translate.py uk ../data/Rinehart_10.txt ../data/Rinehart_10_uk.txt
  ```
  To view in web application, enable the `tag` option (last argument)
  ```shell
  > python driver_translate.py uk ../data/Rinehart_10.txt ../data/Rinehart_10_uk_tag.txt 1
  ```

* Synonym replacement were performed using the embedding models of 50 dimension
  and corresponding document partition size.
  ```shell
  > cd synonyms/
  > python driver_synonyms.py 0 0.2 ../data/Rinehart_10.txt ../data/Rinehart_10_syn_350.txt ../serving/doyle_50dim_350part.bin
  ```
  To view in web application, enable the `tag` option (first argument)
  ```shell
  > python driver_synonyms.py 1 0.2 ../data/Rinehart_10.txt ../data/Rinehart_10_syn_350_tag.txt
  ```

* Test datasets JSON files were created by combining the 10\% perturbed files.
  This script takes multiple text files with corresponding labels and partitions
  them into documents, then shuffles them and exports list of files to a JSON file.
  ```shell
  > cd test_datasets/
  > python driver_create_json.py 350 perturbed_langtranslation_rinehart_350.json ../data/Doyle_10_uk.txt doyle ../data/Christie_10_uk.txt christie ../data/Rinehart_10_uk.txt rinehart
  ```

* There are helper scripts to compute frequency of perturbations for making plots.
  First you need to create JSON file of the original corpus.
  For example, for language translation:
  ```shell
  > cd test_datasets/
  > python driver_create_json.py 350 original_rinehart_350.json ../data/Rinehart_10.txt rinehart
  ```
  Then, process them with `freq` script to the corresponding perturbation,
  ```shell
  > cd lang_translation/
  > python driver_freq_translate.py uk ../test_datasets/original_rinehart_3500.json perturb_rate_langtranslation_rinehart_3500.json
  ```


## Novels and Short Stories

* Selection should have 300K +- 10% words in total.

| Type  | Title                                | Words (N) |
| :---- | :----------------------------------- | :-------: |
| Novel | The Valley of Fear                   | 58,827    |
| Novel | A Study in Scarlet                   | 43,862    |
| Novel | The Sign of the Four                 | 43,705    |
| Novel | The Hound of the Baskervilles        | 59,781    |
| Story | The Boscombe Valley Mystery          | 9,722     |
| Story | The Five Orange Pips                 | 7,388     |
| Story | The Adventure of the Speckled Band   | 9,938     |
| Story | The Adventure of the Cardboard Box   | 8,795     |
| Story | The Musgave Ritual                   | 7,642     |
| Story | The Reigate Squires                  | 7,303     |
| Story | The Adventure of the Dancing Men     | 9,776     |
| Story | The Adventure of the Second Stain    | 9,800     |
| Total | Gensim tokenizer                     | 276,539   |

* Short stories
  * The Adventures of Sherlock Holmes
    * 4 - The Boscombe Valley Mystery
    * 5 - The Five Orange Pips
    * 8 - The Adventure of the Speckled Band
  * Memoirs of Sherlock Holmes (British version)
    * 2 - The Adventure of the Cardboard Box
    * 6 - The Musgave Ritual
    * 7 - The Reigate Squires
  * The Return of Sherlock Holmes
    * 3 - The Adventure of the Dancing Men
    * 13 - The Adventure of the Second Stain


## Pipeline Document
<https://docs.google.com/document/d/1lYdSgOwpMAF2GGBTz4h0kvHQPEfisEoplJDX4_YUQSc/edit?usp=sharing>


## Preprocessing

* Lowercase
* Remove non-alpha symbols
* Lemmatize (NLTK)

### Sentence Segmentation

| Type                    | Sentences (N) |
| :---------------------- | :-----------: |
| NLTK line               | 18,616        |
| NLTK punctuation        | 18,638        |


## Word Embeddings

* word2vec parameters: free choice
* Construct models using embedding sizes: 50 and 300
* For document embeddings, use the entire document (no random words as in paper)
* Unknown tokens are set to a zero vector


## MLP

* MLP parameters: free choice
* For MLP input, average document embeddings into a single vector


## Training, Validation, and Testing Datasets

<!-- Each group creates a training, testing, validation dataset (20% author / 80% other authors). -->
* Data unit - represents a contiguous collection of words that create a "document"
  of the corresponding author. To create a data unit, always start at the beginning
  of a sentence and end when word count is fulfilled.
  * 1/2 page - 350 words
  * 2 page - 1,400 words
  * 5 page - 3,500 words
<!-- * 20% from author's corpus, 80% from other authors - document how you mix it -->
* 90/10 using documents as the data unit
  * Split 90% into 50/25/25
  * 10% for testing
* 90/10, share 10 with other groups to perturb
  * From 10% use 80/20 for defeat dataset


## Adversarial Techniques

* Each group will apply at least 2 perturbations.
  * All groups will do synonyms replacement - approach can differ (free choice)
  * Doyle - US/British English translation
  * Rinehart - contractions or pronouns
  * Christie - undecided
* Apply perturbations to selective data
* The question on how much perturbation to apply to each document will depend
  on the perturbation itself. Some approaches will modify more text than
  others. We suggest to limit the perturbation effect to 20% for each document.
  If a perturbation changes less than 20%, then you can consider all its
  changes. If a perturbation exceeds the 20%, then limit it.
* For synonyms perturbations: 20% upper limit per document
* For second perturbation: up to group's discretion


### Doyle Group Proposed Ideas

* Language translation (USEnglish to British) - Google translate
* Synonym replacement using word vector similarity, part of speech, other model-agnostic qualities
* Change tense - https://github.com/bendichter/tenseflow
* Change singular and plural forms of words, change numbers and text - https://github.com/jazzband/inflect
* Invert text and word order
* Rearrange neighbor sentences
* Introducing typos (letter flipping)

### Reinhart Group Proposed Ideas

* Replace with synonyms, something like this: https://www.tutorialspoint.com/natural\_language\_toolkit/natural\_language\_toolkit\_synonym\_antonym\_replacement.htm or this https://stackoverflow.com/questions/5148377/replacing-synonyms-in-a-corpus-using-wordnet-and-nltk-python
* Augmentation of the novels with generated texts (https://openai.com/blog/gpt-2-1-5b-release/)
* Delete or replace English honorifics (e.g., sir, Mr., Mrs., Miss)
* Obfuscation Mutant-X (https://github.com/asad1996172/Mutant-X)
* Style Neutralization (https://github.com/asad1996172/Obfuscation-Systems/tree/master/Style%20Nueralization%20PAN16)
* Document Simplification (https://github.com/asad1996172/Obfuscation-Systems/tree/master/Document%20Simplification%20PAN17)
* Characters replaced by pronouns
* Contractions

### Edmon's Comments on Proposed Ideas

* Synonyms are good.
* British to US English is OK, but tense change or typos are most likely not.
* Tense can possibly change the meaning of the text, but if done carefully it could be fine.
(Thinks of participles vs. simple past tense, etc. He was in prison/he has been in prison, etc. )
* Character flipping can turn text into gibberish or can alter the meaning. There is no easy way to control it. (E.g. mud/mad, pea/pee, tea/tee, stop/step, and so on ...)
* Plurals and singulars are tricky. He murdered a woman is not the same as he murdered women.
* Re-arranging sentences how? You could consider changing active to passive voice. It is reasonably safe way.
* Changing numbers and text might OK, but you could also squash meaning if done carelessly and automatically.


## Notes from PM meeting on 10/14/20
* All groups will select 4-5 crime novels (same from Project 1) that contain a total of 300K +- 10% tokens. The 4 Doyle's novels used in Project 1 have a total of ~203K words, and given that are no more Holmes' novels, we added 8 short stories.
* We will use 3 data resolutions: 350 words (1/2 page), 1400 words (2 pages), and 3500 words (5 pages). The data units will be selected from the single merged text file by starting at a first word of a sentence and ending at the end of a sentence that results closest in number of words to the data unit but not more. These data units will be non-overlapping.
* Groups will share perturbation ideas. Edmon will decide on a handful to from these to be assigned to groups. The perturbations will vary between groups.
* The goal is to replicate the classification approach presented in assigned paper. We will have 6 w2v models per author (Nx6) and 6 MLP heads.
* We will use two embedding sizes for the vector embeddings: 50 and 300.
* All groups will share text data as follows: Extract only the prose from all novels (no headings, no metadata) and merge together into a single file with no formatting changes except removing empty lines.
