import os
import yaml
import logging
import time
import pandas as pd
import json
from zipfile import ZipFile


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

