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