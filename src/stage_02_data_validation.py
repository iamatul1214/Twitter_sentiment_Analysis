import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from src.utils.data_validation import read_csv, check_null_values,check_binary_classification,check_data_distribution

STAGE = "Data Validation"   ## Name of the stage

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    file_location = config['data']['unzip_data_dir']
    file_name = config['data']['name_to_be_changed']
    columns = config['data']['columns']
    encoding_type = config['data']['encoding_type']
    negative_sentiment = config['data']['negative_sentiment']
    positive_sentiment = config['data']['positive_sentiment']
    

    dataframe = read_csv(file_location=file_location,file_name=file_name,columns=columns,encoding_type=encoding_type)
  #  logging.info(f"dataframe = \n{dataframe}")
    
    null_value_count = check_null_values(dataframe=dataframe)

    if (null_value_count > 0):
         logging.info("The total null values count is more than 0. Hence, need to deal with null values first")

    label_check = check_binary_classification(dataframe= dataframe)

    if(label_check != 2):
        logging.info("The dataset doesn't look like binary classifcation. Hence, exiting the process") 
    else:
        logging.info("The dataset looks like a binary classification. Hence, continuing to next check..")

    data_distribution_ratio = check_data_distribution(dataframe= dataframe,negative_sentiment=negative_sentiment, positive_sentiment=positive_sentiment)

    if data_distribution_ratio > 60 or data_distribution_ratio < 40:
        logging.info("The ratio of data distribution is less than 60:40 hence, failing the pipeline")
    else:
        logging.info("The ratio of the data distribution is under 60:40 range hence, passing the pipeline for next stage")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        #main(config_path=parsed_args.config, params_path=parsed_args.params)
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e