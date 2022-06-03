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

STAGE = "Model testing on unknown dataset"   ## Name of the stage

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


## This file will be only used for testing on some dataset, PLease note that this is not the prediction file. Here we just test the prediction file
## model after testing it from test dataset. This can be termed as model testing on some real dataset But as of now we have used same dataset


    path_to_trained_model = os.path.join(ckpt_dir,trained_model_dir)
    trained_model = tf.keras.models.load_model(path_to_trained_model)

     ## reading the dataframe
    dataframe = read_csv(file_location=file_location,file_name=file_name,columns=columns,encoding_type=encoding_type)

    ##keeping omly important columns and dropping other columns
    filtered_dataframe = keep_important_columns(dataframe= dataframe)

    ## replacing target values
    modified_dataframe = replace_target_values(dataframe= filtered_dataframe)

     ## creating train test splits 
    feature_list, targets = separating_label_feature(dataframe = modified_dataframe)
    x_train, x_test,y_train, y_test = train_test_split_operation(feature_list = feature_list, targets = targets, test_size =train_test_split_ratio,
                                                         random_state =random_state, shuffle = shuffle)


    ## read the train test data from preprocessed dataset folder into to_csv
 #   x_train,x_test,y_train,y_test = read_train_test_split_datasets(file_location=preprocessed_dataset_folder,x_train = "x_train",x_test = "x_test",y_train = "y_train",y_test = "y_test")
    
    ## convert train test subsets into the numpy array
    x_train_numpy,x_test_numpy,y_train_numpy,y_test_numpy = convert_data_into_numpy(x_train = x_train,y_train = y_train, x_test = x_test, y_test=y_test)
    logging.info(f"x_train = {x_train_numpy.shape}, y_train = {y_train_numpy.shape}, x_test = {x_test_numpy.shape}, y_test = {y_test_numpy.shape}")

    test_loss, test_acc = trained_model.evaluate(x_test[:1000],y_test[:1000])
    logging.info(f"test loss: {test_loss}")
    logging.info(f"test accuracy: {test_acc}")

    ## saving the accuracy of testing data

    # create_directories([artifacts])

    # history_obj = trained_model.history
    # plt.plot(history_obj[metrics])
    # plt.plot(history_obj[f'val_{metrics}'])
    # plt.xlabel("Epochs -->")
    # plt.ylabel(f"{metrics} -->")
    # plt.legend([metrics, f'val_{metrics}'])
    # time = datetime.now().strftime("%d_%m_%Y-%I_%M_%S_%p")
    # plot_name=time +'_'+f"testing_{metrics}"+".png"
    # plt.savefig(os.path.join(artifacts,plot_name))
    # logging.info(f"Saved the testing {metrics} graph at location: {artifacts}")

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