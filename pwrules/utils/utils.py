import os
import pickle
import numpy as np
import yaml


def load_pkl(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


def check_path(path):
    if not os.path.exists(path):
        return False
    return True


def nan_mean(values):
    arr = np.array(values)
    if np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
