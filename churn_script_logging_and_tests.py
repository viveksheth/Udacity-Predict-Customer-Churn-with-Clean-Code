'''
This module contains test for churn customer analysis script

Author: Vivek Sheth
Date: January 25, 2024

'''

# Import libaries
import os
import logging
import churn_library as clib
from churn_library import perform_feature_engineering

# Invoke basic logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

######################### UNIT TESTS ##################################


def test_import_data():
    '''
    This module tests import_data() function from the churn_library module
    '''
    # Test if the CSV file is available
    try:
        dataframe = clib.import_data("./data/bank_data.csv")
        logging.info("test_import_data: Data import is successful.")
    except FileNotFoundError as error:
        logging.error("Please try again! cannot find the testing file.")
        raise error

    # Test the dataframe
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info('Data imported: [Rows: %d\tColumns: %d]',
                     dataframe.shape[0], dataframe.shape[1])
    except AssertionError as error:
        logging.error("test_import_data: Import data file is empty.")
        raise error


def test_perform_eda():
    '''
    This module tests perform_eda() function from churn_library module
    '''
    dataframe = clib.import_data("./data/bank_data.csv")

    try:
        clib.perform_eda(dataframe=dataframe)
        logging.info("Testing function perform_eda() is successful")
    except KeyError as error:
        logging.error('Column "%s" not found', error.args[0])
        raise error

    # Assert if `churn_distribution.png` is created or not
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('File %s was found', 'churn_distribution.png')
    except AssertionError as error:
        logging.error('churn_distribution.png file not found')
        raise error

    # Assert if `marital_status_distribution.png` is created or not
    try:
        assert os.path.isfile(
            "./images/eda/marital_status_distribution.png") is True
        logging.info('File %s was found', 'marital_status_distribution.png')
    except AssertionError as error:
        logging.error('marital_status_distribution.png file not found')
        raise error

    # Assert if `customer_age_distribution.png` is created or not
    try:
        assert os.path.isfile(
            "./images/eda/customer_age_distribution.png") is True
        logging.info('File %s was found', 'customer_age_distribution.png')
    except AssertionError as error:
        logging.error('customer_age_distribution.png file not found')
        raise error

    # Assert if `total_transaction_distribution.png` is created or not
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png") is True
        logging.info('File %s was found', 'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error('total_transaction_distribution.png file not found')
        raise err

    # Assert if `data_heatmap.png` is created or not
    try:
        assert os.path.isfile("./images/eda/data_heatmap.png") is True
        logging.info('File %s was found', 'data_heatmap.png')
    except AssertionError as error:
        logging.error('data_heatmap.png file not found')
        raise error


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    two_test_level = False
    try:

        data = clib.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = perform_feature_engineering(data)
        logging.info("Testing of perform_feature_engineering is successful")
        two_test_level = True
    except Exception as error:
        logging.error(
            "Testing perform_feature_engineering failed - Error type %s",
            type(error))

    if two_test_level:
        try:
            assert x_train.shape[0] > 0
            assert x_train.shape[1] > 0
            assert x_test.shape[0] > 0
            assert x_test.shape[1] > 0
            assert y_train.shape[0] > 0
            assert y_test.shape[0] > 0
            logging.info(
                "perform_feature_engineering returned Train / Test set of shape %s %s",
                x_train.shape,
                x_test.shape)

        except AssertionError:
            logging.error(
                "The returned train / test datasets do not appear to have rows and columns")


def test_encoder_helper():
    '''
    Testing module encoder_helper
    '''
    # Load DataFrame
    dataframe = clib.import_data("./data/bank_data.csv")

    # Create `Churn` feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical Features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_df = clib.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns)

        # Data should be the same
        assert encoded_df.equals(dataframe) is True
        logging.info(
            "Testing encoder_helper with %s - is successful", cat_columns)
    except AssertionError as error:
        logging.error(
            "Testing encoder_helper with %s - is failed", cat_columns)
        raise error


def test_train_models():
    '''
    Testing module train_models
    '''
    # Load dataframe
    dataframe = clib.import_data("./data/bank_data.csv")

    # churn feature
    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
    (x_train, x_test, y_train, y_test) = clib.perform_feature_engineering(
        dataframe=dataframe,
        response='Churn')

    # Assert if `logistic_model.pkl` file is present
    try:
        clib.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('File %s was found', 'logistic_model.pkl')
    except AssertionError as error:
        logging.error('Not such file on disk')
        raise error

    # Assert if `rfc_model.pkl` file is present
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s was found', 'rfc_model.pkl')
    except AssertionError as error:
        logging.error('Not such file on disk')
        raise error

    # Assert if `roc_curve_result.png` file is present
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File %s was found', 'roc_curve_result.png')
    except AssertionError as error:
        logging.error('Not such file on disk')
        raise error

    # Assert if `rfc_results.png` file is present
    try:
        assert os.path.isfile('./images/results/rf_results.png') is True
        logging.info('File %s was found', 'rf_results.png')
    except AssertionError as error:
        logging.error('Not such file on disk')
        raise error


if __name__ == "__main__":
    test_import_data()
    test_perform_eda()
    test_perform_feature_engineering()
    test_encoder_helper()
    test_train_models()
