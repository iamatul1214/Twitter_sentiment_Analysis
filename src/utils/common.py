import os
import yaml
import logging
import time
import pandas as pd
import json
from zipfile import ZipFile
import tensorflow as tf


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def unzip_file(source:str, dest:str) ->None:        ## This function will take a source and destination string so that to keep the extracted file to destination
    logging.info("extraction started...")
    with ZipFile(source,"r") as zip_f:
        zip_f.extractall(dest)
    logging.info(f"extracted {source} to the {dest}")  

def rename_file(file_path:str,oldfile_name:str, newfile_name:str)->None:
    try:
        logging.info("Trying to rename the file name")
        if os.path.isfile(os.path.join(file_path,newfile_name)):
            logging.info("File is already renamed. Hence, skipping the renaming")
        else:
            os.rename(os.path.join(file_path,oldfile_name),os.path.join(file_path,newfile_name))
            logging.info(f"Renamed the file from {oldfile_name} to {newfile_name}")
    except exception as e:
        logging.exception(e)

def callbacks(tensorboard_log_dir, checkpoint_dir):

  # tensorboard callbacks - 
  unique_log = time.asctime().replace(" ", "_").replace(":", "")
  tensorboard_log_dir = os.path.join(tensorboard_log_dir, unique_log)
  os.makedirs(tensorboard_log_dir, exist_ok=True)

  tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

  # ckpt callback
  ckpt_file = os.path.join(checkpoint_dir, "model")
  os.makedirs(checkpoint_dir, exist_ok=True)

  ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
      filepath=ckpt_file,
      save_best_only=True
  )

  callback_list = [
                   tb_cb,
                   ckpt_cb
  ]

  return callback_list

