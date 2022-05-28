import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from src.utils.data_validation import read_csv, check_null_values,check_binary_classification,check_data_distribution
from src.utils.data_preprocessing import keep_important_columns, replace_target_values,plot_data_distribution

STAGE = "Data Preprocessing"   ## Name of the stage

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
    eda_plots_dir = config['Eda_artifacts']['plots']
    count_plot_name = config['Eda_artifacts']['count_plot_name']

    ## reading the dataframe
    dataframe = read_csv(file_location=file_location,file_name=file_name,columns=columns,encoding_type=encoding_type)

    ##keeping omly important columns and dropping other columns
    filtered_dataframe = keep_important_columns(dataframe= dataframe)

    ## replacing target values
    modified_dataframe = replace_target_values(dataframe= filtered_dataframe)

    ## storing the count plots on target
    ## making the plot directory if not exists
    create_directories([eda_plots_dir])

    plot_data_distribution(dataframe= modified_dataframe,plot_location=eda_plots_dir,filename=count_plot_name)





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