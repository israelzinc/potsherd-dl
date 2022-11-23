#!/usr/bin/env python3
###############################################################################
# This script is necessary before you perform any train. 
# It reads the training and test folder and create a csv containing the values
# It also creates a label encoder to be used later
#
# PARAMETERS:
# argv[1] is the configuration file. Just use one of the provided ones
#
# RETURN VALUE:
# This script does not return any value, but the models are stored in the models folder
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import sys
import pandas as pd
import numpy as np

from os import path
from glob import glob
from pprint import pprint
from os.path import basename
from lib.utils import Config, Dict
from collections import Counter
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder


def get_targets(f: str) -> str:
    # Get 'target' from '/some/path/of/target_with_id.ext'
    file_name = basename(f).split('.')[0]  # target_with_id
    print(f'File Name: {file_name}')
    return file_name.split('_')[0]  # target


def get_target_paths(input_dir: str) -> tuple:
    target_paths = glob(input_dir)
    targets = [get_targets(f) for f in target_paths]
    return target_paths, targets


def create_df(target_paths: list, targets: list, le: LabelEncoder) -> pd.DataFrame:
    target_labels = le.transform(targets)  # Encode target labels
    df = pd.DataFrame(
        {
            'path': target_paths,
            'class_num': target_labels,
            'class': targets
        }
    )
    return df


def create_fold(df: pd.DataFrame, num_folds: int) -> pd.DataFrame:
    # put a placeholder value
    df["kfold"] = -1

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.class_num.values

    # split the dataset while maintaining the ratio
    kf = model_selection.StratifiedKFold(n_splits=num_folds)

    for fold, (_, test_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[test_idx, "kfold"] = fold

    return df


def create_train_csv(config: Dict, label_encoder_path: str):
    target_paths, targets = get_target_paths(config.dir)

    # Create Label Encoders
    le = LabelEncoder()

    # Load the label encoder if it is exist
    if path.exists(label_encoder_path):
        le.classes_ = np.load(label_encoder_path)
    else:
        le.fit(list(set(targets)))

    df = create_df(target_paths, targets, le)
    df = create_fold(df, config.num_folds)

    # save the df
    df.to_csv(config.csv, index=False)

    # Load the label encoder if it is not exist
    if not path.exists(label_encoder_path):
        np.save(label_encoder_path, le.classes_)

    # Debug
    pprint(dict(Counter(targets)))


def create_test_csv(config: Dict, label_encoder_path: str):
    target_paths, targets = get_target_paths(config.dir)

    # Create Label Encoders
    le = LabelEncoder()
    le.classes_ = np.load(label_encoder_path)

    df = create_df(target_paths, targets, le)

    # save the df
    df.to_csv(config.csv, index=False)

    # Debug
    pprint(dict(Counter(targets)))


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Path[s] of configuration file[s] needed.")
        exit(1)

    for arg in sys.argv[1:]:
        print(f'Loading configuration "{arg}"')
        config = Config.load_json(arg)
        datasets = config.datasets  # Get dataset configuration only (for simplification)

        print('Creating Train CSV')
        create_train_csv(datasets.train, datasets.label_encoder)
        print('Creating Train CSV')
        create_test_csv(datasets.test, datasets.label_encoder)
        print('Creating Jomon CSV')
        create_test_csv(datasets.real, datasets.label_encoder)
