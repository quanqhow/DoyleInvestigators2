# Doyle Investigators - Vector Embeddings and Adversarial Analysis

This project constructs a binary classifier for Sir Arthur Conan Doyle using
a dataset of Sherlock Holmes novels and short stories.


## Novels and Short Stories

* Selection should have 300K +- 10% words in total.

| Type  | Title                                | Words (N) |
| :---- | :----------------------------------- | :-------: |
| Novel | The Hound of the Baskervilles        | 59,170    |
| Novel | The Sign of the Four                 | 43,050    |
| Novel | A Study in Scarlet                   | 43,296    |
| Novel | The Valley of Fear                   | 57,688    |
| Story | The Boscombe Valley Mystery          | 9,615     |
| Story | The Adventure of the Speckled Band   | 9,805     |
| Story | The Five Orange Pips                 | 7,314     |
| Story | The Adventure of the Cardboard Box   | 8,680     |
| Story | The Musgave Ritual                   | 7,550     |
| Story | The Reigate Squires                  | 7,164     |
| Story | The Adventure of the Dancing Men     | 9,624     |
| Story | The Adventure of the Second Stain    | 9,617     |
| Total |                                      | 272,573   |

* Short stories
  * The Adventures of Sherlock Holmes
    * The Boscombe Valley Mystery
    * The Adventure of the Speckled Band
    * The Five Orange Pips
  * Memoirs of Sherlock Holmes
    * The Adventure of the Cardboard Box
    * The Musgave Ritual
    * The Reigate Squires
  * The Return of Sherlock Holmes
    * The Adventure of the Dancing Men
    * The Adventure of the Second Stain


## Pipeline Document
<https://docs.google.com/document/d/1lYdSgOwpMAF2GGBTz4h0kvHQPEfisEoplJDX4_YUQSc/edit?usp=sharing>


## Preprocessing

* Lowercase
* Remove non-alpha symbols
* Lemmatize (NLTK)


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
* Data unit
  * 1/2 page - 350 words
  * 2 page - 1,400 words
  * 5 page - 3,500 words
<!-- * 20% from author's corpus, 80% from other authors - document how you mix it -->
* 90/10 using documents as the data unit
  * Split 90% into 50/25/25
  * 10% for testing
* 90/10, share 10 with other groups to perturb
  * From 10% use 80/20 for defeat dataset


### Chris Example: Defeat Dataset

Example: You have 1000 documents. 100 of them form the validation dataset. You send those to the other team. They find 400 other documents, for a total validation/defeat dataset of 500 documents.


<!-- ### Defeat Dataset -->
<!--  -->
<!-- Each group creates a validation and defeat dataset to give to other groups (20/80%). -->
<!-- * 20% from author's corpus, 80% from other authors - document how you mix it -->
<!-- * Alter the 20% input - do not change meaning nor semantics of original text -->
<!-- * Select adversarial techniques -->
<!-- * Document how you construct the defeat dataset, and absolutely document model performance. -->


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
