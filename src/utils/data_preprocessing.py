import os
import yaml
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

## remove not mandatory columns from the dataframe

def keep_important_columns(dataframe):
    try:
        logging.info("Trying to remove all the non important columns and just keep target and tweets column inthe dataframe")
        filtered_dataframe = dataframe[['target','tweet_text']]
        logging.info(f"filtered dataset head looks like \n {filtered_dataframe.head()}")
        return filtered_dataframe 
    except Exception as e:
        logging.exception(e)

def replace_target_values(dataframe):
    try:
        logging.info("Trying to replace the the value of 4 for positive sentiment to 1")
        dataframe['target'] = dataframe['target'].replace(4,1)
        logging.info("successfully replace_target value from 4 to 1")
        return dataframe
    except Exception as e:
        logging.exception(e)

def plot_data_distribution(dataframe,plot_location,filename):
    try:
        logging.info("Trying to plot the data distribution with count plot and store it in the location : {plot_location}")
        sns.countplot(x='target', data=dataframe)
        time = datetime.now().strftime("%d_%m_%Y-%I_%M_%S_%p")
        plot_name=time +'_'+filename+".png"
        plt.savefig(os.path.join(plot_location,plot_name))
        logging.info(f"Saved the count plot figure in the location: {plot_location} with name: {plot_name}")
    except Exception as e:
        logging.exception(e)

def separating_label_feature(dataframe):
    try:
        logging.info("Trying to store tweet text and target values differently")
        X = dataframe.tweet_text
        Y = dataframe.target
        logging.info(f"The x and y data is successfully separated: {X} and {Y}")
        return X,Y

    except Exception as e:
        logging.exception(e)

def train_test_split_operation(feature_list, targets, test_size, random_state,shuffle=True):
    try:
        logging.info("Trying to create the train test split")
        x_train,x_test,y_train,y_test = train_test_split(feature_list, targets, test_size=test_size, random_state = random_state, shuffle=shuffle)
        logging.info(f"The train test split happened properly with ratio: {test_size} and shape of x_train: {x_train.shape}, y_train:{y_train.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.exception(e)

def store_preprocessed_dataset(x_train, y_train, x_test, y_test, folder_location):
    try:    
        logging.info(f"Trying to delete the older stored datasets into the location : {folder_location}")
        total_files = os.listdir(folder_location)
        if len(total_files) > 0:
            for f in os.listdir(folder_location):                
                os.remove(os.path.join(folder_location, f))
                logging.info(f"Deleted {f} stored datasets from the location : {folder_location}")
        else:
            logging.info(f"There are no preprocessed files present already.")
        
        x_train.to_csv(os.path.join(folder_location,"x_train.csv"))
        y_train.to_csv(os.path.join(folder_location,"y_train.csv"))
        x_test.to_csv(os.path.join(folder_location,"x_test.csv"))
        y_test.to_csv(os.path.join(folder_location,"y_test.csv"))
        logging.info("Saved the latest dataset to the location: {folder_location}")

    except Exception as e:
        logging.exception(e)

def convert_data_into_numpy(x_train, y_train, x_test, y_test):
    try:
        logging.info("Trying to convert the dataset into numpy and flatten them")
        x_train = np.array([x_train]).flatten()
        x_test = np.array([x_test]).flatten()
        y_train = np.array([y_train]).flatten()
        y_test = np.array([y_test]).flatten()
        logging.info("Converted successfully the dataset into numpy array and flattened it")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.exception(e)

def convert_numpy_dataset_to_tensors(x_train,y_train,x_test,y_test):
    try:
        logging.info("Trying to convert the numpy datasets into tensors")
        x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train)
        y_test_dataset = tf.data.Dataset.from_tensor_slices(y_test)
        logging.info("Successfully converted the numpy datasets into tensors")
        return x_train_dataset, x_test_dataset,y_train_dataset, y_test_dataset
    except Exception as e:
        logging.exception(e)