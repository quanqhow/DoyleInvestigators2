Shared drive
https://drive.google.com/drive/folders/12-BZxv-1_9dL5PbD55n6dEpIBSMJhVDM?usp=sharing


Colab notebook
https://colab.research.google.com/drive/1QsaMXOws4i-iFGE_5PFxouTMA5GE10jp?usp=sharing


Notes:
* w2v models have already been run for each author, see `models/`. You can check
  this process in `Section 2 w2v models: sentence, and 50, 300 emb\_size`.
* The main section that you would have to access and modify is `Classifier`.
  * In the subsection `Data from each author`, you would have to create the
    training and testing dataset for each author, there's the example for
    Rinehart. There are three cells which are for each data unit. You would
    have to replicate those three cells with the respective 90\% of positive
    authors and 50 from the other two.
  * In the subsection `Data Perturbation`, only Doyle will have to create a
    directory and add the JSON files there, because I created the one for
    Christie (`data/perturbed_data_christie`). Then, you would have to replicate
    the cell `Read JSON files to df` with the right paths.
  * In the subsection `Training MLP model`, there are three cells where we train
    for each data unit and embedding size. You would have to replicate those
    functions with the right w2v model and data.
  * In the subsection `Testing the MLP model`, there are six cells where we test
    for original data, perturbation synonym, and perturbation extra (for each
    embedding size and data unit). You would have to create those same cells with
    the correct data and w2v model.
