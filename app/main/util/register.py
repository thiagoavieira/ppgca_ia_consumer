# ----------------------------------------------------------------------------------------------------------------------
#   Copyright PPGCA - USP RP.
#   This software is confidential and property of PPGCA - USP RP.
#   Your distribution or disclosure of your content is not permitted without express permission from PPGCA - USP RP.
#   This file contains proprietary information.
# ----------------------------------------------------------------------------------------------------------------------
import os
import yaml
from app.main.processor import *

def init_processors():
    """
    Initialize the processors.
    Returns:
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_directory, '..', 'configuration', 'config.yaml')
    config_file = os.path.normpath(config_file)

    with open(config_file) as file:
        processors_config = yaml.load(file, Loader=yaml.FullLoader)

    processors = []

    if "processors" in processors_config:
        for name, conf in processors_config["processors"].items():
            processor = globals()[name](conf)
            processors.append(processor)

    return processors

