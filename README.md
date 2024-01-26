# Udacity-Predict-Customer-Churn-with-Clean-Code

## Project Description
This is the first project of Udacity's Machine Learning DevOps Engineer Nanodegree.
The project objective is to produce production-ready clean code using best practices.
The project itself aims at predicting customer churn for banking customers. This is a classification problem.
The project proposes the following approach:
- Load and explore the dataset composed of over 10k samples (EDA)
- Prepare data for training (feature engineering resulting into 19 features)
- Train two classification models (sklearn random forest and logistic regression)
- Identify most important features influencing the predictions and visualize their impact using SHAP library
- Save best models with their performance metrics

## Files and data description
#### Overview of the files and data present in the root directory
The project is organized with the following directory architecture:
- Folders
    - Data      
        - eda       --> contains output of the data exploration
        - results   --> contains the dataset in csv format
    - images        --> contains model scores, confusion matrix, ROC curve
    - models        --> contains saved models in .pkl format
    - logs          --> log generated druing testing of library.py file

- project files 
    - churn_library.py
    - churn_notebook.ipnyb
    - requirements_py3.9.txt

- test files 
    - churn_script_logging_and_test.py


## Running Files
