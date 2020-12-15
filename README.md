# Instructions to run CPS803 F2020 Group 21's Project
To facilitate the phased development and implementation of this project, several separate Python programs were created for different steps of the process and various models used. They have been consolidated to call from a single program for ease of use, but still retain their main() function definitions and could be called separately if needed.

# The following Python libraries need to be installed to run the project:
- numpy
- scikit learn
- nltk (including nltk.corpus and nltk.punkt modules)
- matplotlib

# Retrieving and preparing the dataset
1. Download [dataset](https://www.kaggle.com/vetrirah/janatahack-independence-day-2020-ml-hackathon/download)
2. Delete the files `sample_submission.csv` and `test.csv`
3. Create a directory called `dataset` within the main project directory
4. Rename the file `train.csv` to `original_data.csv` and place it in the `dataset` directory
5. Run `python3 splitdataset.py` from the main project directory
  - this preprocesses the dataset, splitting it into training and testing data with separate files for abstracts and labels. It also creates a 1000-entry subset of the data called tinydataset and preprocesses it in the same way

# Running project to preprocess text data, fit and predict with implemented models, generate evaluation reports and graphic plots
1. Create a directory called `output` within the main project directory
2. Run `python3 project_main.py` from the main project directory
- Outputs will be saved to the `output` directory
- Outputs will include .txt report files with accuracy, rates and precision metrics for the two models (2 files per model; separate files for predicting on training and testing data)
- Outputs will also include 3 .jpg files containing data plots

# helper_programs
- There are several Python programs in the `helper_programs` directory
- They were used to accomplish small tasks throughout the development process which do not need to be repeated to implement fitting and testing the models, but may be useful if the dataset is changed.

# deprecated
- There are several Python programs in the `deprecated` directory
- They contain code which was either with a newer version in a different file or code which ended up not being used within the scope of this project


