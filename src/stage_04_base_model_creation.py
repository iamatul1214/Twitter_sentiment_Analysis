import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from src.utils.data_preprocessing import keep_important_columns, replace_target_values,plot_data_distribution,separating_label_feature,train_test_split_operation,store_preprocessed_dataset,convert_data_into_numpy, convert_numpy_dataset_to_tensors
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
    preprocessed_dataset_folder = config['data']['preprocessed_data_folder']
    train_test_split_ratio = config['model_config']['train_test_split']
    random_state = config['model_config']['random_state']
    shuffle = config['model_config']['shuffle']
    

    
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

    ## converting the numpy arrays into tensors
    x_train_tensor,x_test_tensor,y_train_tensor,y_test_tensor = convert_numpy_dataset_to_tensors(x_train = x_train_numpy,y_train = y_train_numpy,
                                                                                                    x_test = x_test_numpy,y_test = y_test_numpy)
    logging.info(f"The shape of x_train_tensor = {len(x_train_tensor)}, y_train_tensor = {len(y_train_tensor)}")

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