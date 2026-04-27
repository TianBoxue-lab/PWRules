import os
import pickle


def load_pkl(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


def check_path(path):
    if not os.path.exists(path):
        return False
    return True
