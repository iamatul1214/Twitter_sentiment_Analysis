## This file is getting the data from source

import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, unzip_file
import random
import urllib.request as req
from src.utils.data_mgmt import validate_image
import pandas as pd

STAGE="Get_Data"    ## Name of the stage

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    url = config["data"]["url"]
    local_dir=config["data"]["local_dir"]