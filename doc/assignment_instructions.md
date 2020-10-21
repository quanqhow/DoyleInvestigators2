For this assignment, you will techniques we covered so far, specifically word embeddings, to try to identify the known author (Christie, Doyle, Rinehart) of the crime novels available on the Project Gutenberg from the supplied texts. Furthermore, you will explore the strengths and weaknesses of this approach, by applying the adversarial exploitation approaches to try to defeat those modes.  This is, again, an open-ended approach and, more importantly, novel, so we will work to capture the results from the projects into one single research paper and try to publish it.

I will be assisting you along the way, but you will need to apply significant effort to complete the project, and to stay focused on the required deliverables.
Tasks

For the assignment, you will be doing two specific analytic tasks:

Part 1: You will develop a classifier that can correctly classify the piece of writing as being written by the author you analyzed in Project 1, or not. You will use this https://utk.instructure.com/courses/110610/files/6859225/download?wrap=1. Preview the document as a foundation, and you will try to implement their approach. This does not mean that it is the best approach, but it should serve as an inspiration and a starting point for your implementation.

Part 2: You will develop a testing dataset for the other team. This testing dataset will have two parts:

1. a mixture of short excerpts (0.5 pages (350 words), 2 pages (750 words)  and 5 pages (1750 words)) from the author they have to classify, interspersed with the writings of any other crime authors available on the project Guttenberg, and

2. an adversarial test set. You will take short excerpts, and perturb them so that a) they retain linguistic qualities of the original text (they are readable, have proper English form), but b) they are perturbed to confuse the word embedding-based model and other more-primitive techniques that you think will be used (n-grams, hard-coded rules).
