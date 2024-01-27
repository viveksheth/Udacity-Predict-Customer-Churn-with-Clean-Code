# Udacity-Predict-Customer-Churn-with-Clean-Code

## Project Description
The objective of this project is to produce production-ready clean code using best practices. The project includes a python package for a machine learning project that follows coding best practices (PEP8) and best practice of software development such as modules, documentations and unit tests. 

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
## Setup your enviornment 

Python version: 3.9.x 

#### 1. Clone the repository. 

#### 2. Install libraries: 

```python3 -m pip install -r requirements_py3.9.txt```

#### 3. Install linter and auto-formatter: 

```pip3 install pylint autopep8``` 

#### 4. Run code 

To run project script: ```python3 churn_library.py```

To run unit tests: ```python3 churn_script_logging_and_tests.py```







