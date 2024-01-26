'''
This module contains different funcations to perform churn analysis on bank customer data

Author: Vivek Sheth
Date: January 16, 2024

'''

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth, index_col=0)

    # Encode Churn dependent variable : 0 = Did not churned ; 1 = Churned
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop redudant Attrition_Flag variable (replaced by Churn response
    # variable)
    # dataframe.drop('Attrition_Flag', axis=1, inplace=True)

    # Drop variable not relevant for the prediction model
    # dataframe.drop('CLIENTNUM', axis=1, inplace=True)

    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder

    input:
                    dataframe: pandas dataframe

    output:
                    None
    '''
 
    # Analyze categorical features and plot distribution
    plt.figure(figsize=(20,10))
    dataframe['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Customer age distribution  
    plt.figure(figsize=(20,10))
    dataframe['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'],kde=True);
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, linewidths=3, cmap='Dark2_r')
    plt.savefig(fname='./images/eda/data_heatmap.png')

def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
                    dataframe: pandas dataframe
                    category_lst: list of columns that contain categorical features
                    response: string of response name [optional argument that
                    could be used for naming variables or index y column]

    output:
                    dataframe: pandas dataframe with new columns for
    '''
    category_groups = []
    for category in category_lst:
        category_groups = dataframe.groupby(category).mean()[response]
        new_feature = category + '_' + response
        dataframe[new_feature] = dataframe[category].apply(
            lambda x: category_groups.loc[x])

    # Drop the obsolete categorical features of the category_lst
    dataframe.drop(category_lst, axis=1, inplace=True)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    Converts remaining categorical using one-hot encoding adding the response
    str prefix to new columns Then generate train and test datasets

    input:
                      dataframe: pandas dataframe
                      response: string of response name [optional argument that
                      could be used for naming variables or index y column]

    output:
                      x_train: X training data
                      x_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''

    # Collect categorical features to be encoded
    cat_columns = dataframe.select_dtypes(include='object').columns.tolist()

    # Encode categorical features using mean of response variable on category
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')
    # Alternative to the encodign approach above - Not used here
    # convert categorical features to dummy variable
    #df = pd.get_dummies(df, columns=cat_columns, drop_first=True, prefix=response)

    y = dataframe[response]
    X = dataframe.drop(response, axis=1)
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display figure
    # plt.show()
    # plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder using plot_classification_report
    helper function

    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''

    # RandomForestClassifier 
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')

    # LogisticRegression 
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, features, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''

     # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Sorted feature importances
    names = [features.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(25, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features.shape[1]), importances[indices])

    # x-axis labels
    plt.xticks(range(features.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')


def confusion_matrix(model, model_name, X_test, y_test):
    '''
        Display confusion matrix of a model on test data

        input:
                        model: trained model
            X_test: X testing data
                        y_test: y testing data
        output:
                        None
        '''
    class_names = ['Not Churned', 'Churned']
    plt.figure(figsize=(15, 5))
    ax_plot = plt.gca()
    plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          display_labels=class_names,
                          cmap=plt.cm.Blues,
                          xticks_rotation='horizontal',
                          colorbar=False,
                          ax=ax_plot)
    # Hide grid lines
    ax_plot.grid(False)
    plt.title(f'{model_name} Confusion Matrix on test data')
    plt.savefig(
        os.path.join(
            "./images/results",
            f'{model_name}_Confusion_Matrix'),
        bbox_inches='tight')
    # # plt.show()
    # plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
                      x_train: X training data
                      x_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''

    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': [ 'sqrt'],
                  'max_depth' : [4, 5, 100],
                  'criterion' :['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf  = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr  = lrc.predict(X_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)                          
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)       
    plt.savefig(fname='./images/results/roc_curve_result.png')
    #plt.show()

    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr,  y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            features=X_test,
                            output_pth='./images/results/')

if __name__ == "__main__":
    dataset = import_data("./data/bank_data.csv")
    print('Dataset is loaded successfully.')
    perform_eda(dataset)
    x_train_model, x_test_data, y_train_model, y_test_data = perform_feature_engineering(
        dataset, response='Churn')
    print('Training data...')
    train_models(x_train_model, x_test_data,  y_train_model, y_test_data)
    print('Model training completed')
          
