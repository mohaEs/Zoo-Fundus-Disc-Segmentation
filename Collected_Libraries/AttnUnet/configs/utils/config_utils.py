# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""
import argparse
import json

import os
from bunch import Bunch

from Collected_Libraries.AttnUnet.configs.utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    change json file to dictionary
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  

    config = Bunch(config_dict)  

    return config, config_dict


def process_config(json_file):
    """
    solve json file
    :param json_file: 
    :return: 
    """
    config, _ = get_config_from_json(json_file)

    return config




