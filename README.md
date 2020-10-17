# Doyle Investigators - Vector Embeddings and Adversarial Analysis

## Document for notes
https://docs.google.com/document/d/1lYdSgOwpMAF2GGBTz4h0kvHQPEfisEoplJDX4\_YUQSc/edit?usp=sharing


## Notes for PM meeting on 10/14/20
* All groups will select 4-5 crime novels (same from Project 1) that contain a total of 300K +- 10% tokens. The 4 Doyle's novels used in Project 1 have a total of ~203K words, so we need to select an additional Sherlock Holmes' "crime" novel.
* We will use 3 data resolutions: 350 words (1/2 page), 1400 words (2 pages), and 3500 words (5 pages). The data units will be selected from the single merged text file by starting at a first word of a sentence and ending at the end of a sentence that results closest in number of words to the data unit but not more. These data units will be non-overlapping.
* Groups will share perturbation ideas. Edmon will decide on a handful to from these to be assigned to groups. The perturbations will vary between groups.
* The goal is to replicate the classification approach presented in assigned paper. We will have 6 w2v models per author (Nx6) and 6 MLP heads.
* We will use two embedding sizes for the vector embeddings: 50 and 300.
* All groups will share text data as follows: Extract only the prose from all novels (no headings, no metadata) and merge together into a single file with no formatting changes except removing empty lines.


## Perturbation Ideas

### Doyle Group

* Language translation (USEnglish to British) - Google translate
* Synonym replacement using word vector similarity, part of speech, other model-agnostic qualities
* Change tense - https://github.com/bendichter/tenseflow
* Change singular and plural forms of words, change numbers and text - https://github.com/jazzband/inflect
* Invert text and word order
* Rearrange neighbor sentences
* Introducing typos (letter flipping)

### Reinhart Group

* Replace with synonyms, something like this: https://www.tutorialspoint.com/natural\_language\_toolkit/natural\_language\_toolkit\_synonym\_antonym\_replacement.htm or this https://stackoverflow.com/questions/5148377/replacing-synonyms-in-a-corpus-using-wordnet-and-nltk-python
* Augmentation of the novels with generated texts (https://openai.com/blog/gpt-2-1-5b-release/)
* Delete or replace English honorifics (e.g., sir, Mr., Mrs., Miss)
* Obfuscation Mutant-X (https://github.com/asad1996172/Mutant-X)
* Style Neutralization (https://github.com/asad1996172/Obfuscation-Systems/tree/master/Style%20Nueralization%20PAN16)
* Document Simplification (https://github.com/asad1996172/Obfuscation-Systems/tree/master/Document%20Simplification%20PAN17)
* Characters replaced by pronouns
* Contractions

### Edmon's Comments

* Synonyms are good.
* British to US English is OK, but tense change or typos are most likely not.
* Tense can possibly change the meaning of the text, but if done carefully it could be fine.
(Thinks of participles vs. simple past tense, etc. He was in prison/he has been in prison, etc. )
* Character flipping can turn text into gibberish or can alter the meaning. There is no easy way to control it. (E.g. mud/mad, pea/pee, tea/tee, stop/step, and so on ...)
* Plurals and singulars are tricky. He murdered a woman is not the same as he murdered women.
* Re-arranging sentences how? You could consider changing active to passive voice. It is reasonably safe way.
* Changing numbers and text might OK, but you could also squash meaning if done carelessly and automatically.
