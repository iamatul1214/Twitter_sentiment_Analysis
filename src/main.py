import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import mlflow


STAGE = "MAIN"                      ## << stage name 

create_directories(["logs"])
with open(os.path.join("logs","running_logs.log"), "w") as f:
    f.write(" ")

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main():
    with mlflow.start_run() as run:
        pass
        mlflow.run(".", "get_data", use_conda=False)
  #     mlflow.run(".","get_data", ""parameters={},use_conda=False)      We can use paramters as well
        mlflow.run(".","data_validation",use_conda=False)
        mlflow.run(".","data_preprocessing",use_conda=False)
        mlflow.run(".","base_model_creation",use_conda=False)
 #       mlflow.run(".","model_training",use_conda=False)   ##skipping the model training as it takes too long
        mlflow.run(".","model_testing",use_conda=False)
        mlflow.run(".","model_prediction", use_conda=False)



if __name__ == '__main__':

    try:
        logging.info("\n**********************************************************************************************************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
