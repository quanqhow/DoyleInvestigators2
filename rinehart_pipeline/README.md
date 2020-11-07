Shared drive
https://drive.google.com/drive/folders/12-BZxv-1_9dL5PbD55n6dEpIBSMJhVDM?usp=sharing


Colab notebook
https://colab.research.google.com/drive/1QsaMXOws4i-iFGE_5PFxouTMA5GE10jp?usp=sharing


Notes:
* w2v models have already been run for each author, see `models/`. You can check
  this process in `Section 2 w2v models: sentence, and 50, 300 emb_size`.
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


# Setup Instructions

* Created a Shared Drive in your Google Drive, named `ECE-692 Rinehart Investigators `
  (NOTE: there is an extra whitespace at the end of folder name).
* Open Rinehart's shared drive, https://drive.google.com/drive/folders/12-BZxv-1_9dL5PbD55n6dEpIBSMJhVDM?usp=sharing
  and in that drives menu (a drop down menu with a small downward arrow next to name `Project 2`), select `Add shortcut to Drive`, and add a link to the Shared Drive
  created in previous step.
* Open Colab notebook, `project2_doyle.ipynb` and after mounting the `/content/drive`
  you should have available the following path `/content/drive/Shared drives/ECE-692 Rinehart Investigators /Project 2/`. This is needed to load the word2vec models
  already available in `/.../Project 2/models` and the notebook is set up to
  load them from this path.
* Also, you need to 2 additional folders in your personal Google drive:
  * One folder for saving MLP models, the code currently saves them at `/content/drive/My Drive/clf_doyle/`.
  * One folder for staging perturbed testing data, the code currently uses `/content/drive/My Drive/perturbed_data_doyle`. In this folder you need to upload 6 files
    from our GitHub repo in `test_datasets/perturbed_xxx_doyle_xxx.json`.
* Running the notebook is fairly straightforward, each section has a meaningful
  title.
  * Maofeng will run models using embedding size of 50 dimensions.
  * Fabian will run models using embedding size of 300 dimensions.
  * Each member should fill out Doyle rows in Table 4 and Table 5 in Overleaf
    paper with the corresponding accuracies. The accuracies are printed in
    notebook after testing is completed along with its confusion matrix.
