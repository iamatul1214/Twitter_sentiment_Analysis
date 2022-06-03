import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import random
import numpy as np
from src.utils.common import callbacks
from src.utils.data_preprocessing import keep_important_columns, replace_target_values,plot_data_distribution,separating_label_feature,train_test_split_operation,store_preprocessed_dataset,convert_data_into_numpy, convert_numpy_dataset_to_tensors
from src.utils.data_validation import read_csv, check_null_values,check_binary_classification,check_data_distribution
import matplotlib.pyplot as plt

STAGE = "Model prediction on sample text"   ## Name of the stage

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
    preprocessed_dataset_folder = config['data']['preprocessed_data_folder']
    train_test_split_ratio = config['model_config']['train_test_split']
    random_state = config['model_config']['random_state']
    shuffle = config['model_config']['shuffle']
    vocab_size = config['model_config']['VOCAB_SIZE']
    batch_size = config['model_config']['BATCH_SIZE']
    base_model_dir = config['artifacts']['BASE_MODEL_DIR']
    base_model_name = config['artifacts']['BASE_MODEL_NAME']
    output_dim = config['model_config']['OUTPUT_DIM']
    tnsbrd_log_dir = config['artifacts']['TENSORBOARD_ROOT_LOG_DIR']
    ckpt_dir = config['artifacts']['CHECKPOINT_DIR']
    path_to_model = config['artifacts']['BASE_MODEL_DIR']
    epochs = config['model_config']['EPOCHS']
    trained_model_dir = config['artifacts']['TRAINED_MODEL_DIR']
    artifacts = config['artifacts']['ARTIFACTS_DIR']
    metrics = config['model_config']['METRICS']
    sample_text1 = config['Prediction_data']['sample_text_1']
    sample_text2 = config['Prediction_data']['sample_text_2']


    path_to_trained_model = os.path.join(ckpt_dir,trained_model_dir)
    trained_model = tf.keras.models.load_model(path_to_trained_model)

    predictions = trained_model.predict(np.array([sample_text2]))
    score = predictions[0][0]

    if score > 0 :
        logging.info(f"result: positive sentiment with score: {score}")
    else:
        logging.info(f"result: negetive sentiment with score: {score}")



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