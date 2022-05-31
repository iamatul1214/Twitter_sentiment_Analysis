import os
import yaml
import pandas as pd
import numpy as np
import time
import logging
import json

def read_csv(file_location:str,file_name:str,columns:str,encoding_type:str):
    try:
        logging.info(f"Trying to read the csv file from the location: {file_location}")
        data = pd.read_csv(os.path.join(file_location,file_name), names=columns, encoding=encoding_type)
        logging.info(f"glimpse of the read dataset from {file_location} \n {data.head(5)}")
        return data
    except Exception as e:
        logging.exception(e)

def check_null_values(dataframe):
    try:
        logging.info(f"Trying to check the null values in the dataset")
        null_values_info_series = np.sum(dataframe.isnull())
        logging.info(f"The null values in the dataset is as follows \n {null_values_info_series}")
        total_null_values = np.sum(dataframe.isnull().any(axis=1))
        logging.info(f"The overall null values in the dataset = {total_null_values}")
        return int(total_null_values)
    except Exception as e:
        logging.exception(e)

def check_binary_classification(dataframe):
    try:
        logging.info("Checking binary classification of targets column in the dataset")
        unique_labels = dataframe['target'].nunique()
        logging.info(f"The dataset has {unique_labels} unique labels")
        return int(unique_labels)
    except Exception as e:
        logging.exception(e)

def check_data_distribution(dataframe, negative_sentiment=0, positive_sentiment=4) -> float:
    try:
        logging.info("Trying to check the distribution of the dataset")
        negative = dataframe.loc[dataframe.target==negative_sentiment].count()['target']
        positive = dataframe.loc[dataframe.target==positive_sentiment].count()['target'] 
        total_size = dataframe.target.size
        ratio = (positive/total_size) * 100
        logging.info(f"The percentage of the data distribution of postive class is {ratio}%")
        return ratio
    except Exception as e:
        logging.exception(e)

def read_train_test_split_datasets(file_location,x_train,x_test,y_train,y_test):
    try:
        logging.info(f"Reading the x_train,x_test, y_train,y_test from location: {file_location}")
        filetype = ".csv"
        x_train = pd.read_csv(os.path.join(file_location,x_train+filetype))
        x_test = pd.read_csv(os.path.join(file_location,x_test+filetype))
        y_train = pd.read_csv(os.path.join(file_location,y_train+filetype))
        y_test = pd.read_csv(os.path.join(file_location,y_test+filetype))
        logging.info(f"successfully read the x_train,x_test,y_train,y_test from location: {file_location}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.exception(e)