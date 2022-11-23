import json
import numpy as np
import pandas as pd


class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):

    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Config.__load__(value)
        return result

    @staticmethod
    def load_list(data: list):
        result = [Config.__load__(item) for item in data]
        return result

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))
        return result


def confusion_matrix(y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_labels = np.unique(y)
    y_pred_labels = np.unique(y_pred)

    matrix = pd.DataFrame(np.zeros((len(y_labels), len(y_pred_labels))),
                          index=y_labels, columns=y_pred_labels, dtype=int)

    for c, p in zip(y, y_pred):
        matrix.loc[c, p] += 1

    return matrix
