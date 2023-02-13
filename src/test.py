#!/usr/bin/env python3

import sys
import torch
import pandas as pd
import numpy as np
import albumentations as A

from typing import List, Tuple
import torch.nn as nn

from pprint import pprint
from sklearn.metrics import classification_report
from os.path import join
from lib import engine, models
from lib.dataset import Dataset
from lib.utils import Dict, Config, print_confusion_matrix, load_model, get_test_folds, get_loader_with_augs
from sklearn.preprocessing import LabelEncoder

def print_report(config_files, final_preds):
    conf = config_files[0]
    datasets_csv = conf.datasets.test.csv
    df = pd.read_csv(datasets_csv,nrows=21)
    print("classification_report")
    print(classification_report(df.class_num.values, final_preds))
    print("Confusion Matrix (row/col = act/pred)")
    print(print_confusion_matrix(df.class_num.values, final_preds))

    # Create Label Encoders
    le = LabelEncoder()
    le.classes_ = np.load(conf.datasets.label_encoder)
    labels_num = list(range(conf.model.n_classes))
    labels = le.inverse_transform(labels_num)
    print("Labels")
    pprint(list(zip(labels_num, labels)))

def predict_for_loader(model,loader, conf):    
    predictions = engine.predict(model, loader, conf.model.device)
    preds = []
    for vp in predictions:
            preds.extend(vp)  
    p = np.vstack((predictions))
    return p

def predict_tta(config_file: Dict, test_images: List[str], model: nn.Module):        
    
    fixed_augs = [    
            A.SmallestMaxSize(config_file.model.size),
            A.CenterCrop(config_file.model.size[0], config_file.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]    

    tta_aug = [
        A.VerticalFlip(always_apply=True),
        A.HorizontalFlip(always_apply=True),        
        A.ShiftScaleRotate(always_apply=True),        
    ]
    
    final_predictions = []

    #Augmented Predictions
    for aug in tta_aug:
        augmentation_list = fixed_augs + [aug]
        loader = get_loader_with_augs(config_file, augmentation_list, test_images)
        preds = predict_for_loader(model,loader, config_file)        
        final_predictions.append(preds)          
    return sum(final_predictions)/(len(tta_aug))

def predict(conf, test_images, model):
    test_targets = np.zeros(len(test_images), dtype=np.int_)

    test_aug = A.Compose(
        [
            A.SmallestMaxSize(conf.model.size),
            A.CenterCrop(conf.model.size[0], conf.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]
    )

    test_dataset = Dataset(
        image_paths=test_images,
        targets=test_targets,
        augmentations=test_aug,
        channel_first=True,
        torgb=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.DataLoader.batch_size,
        shuffle=False,
        num_workers=conf.DataLoader.num_workers
    )

    predictions = engine.predict(model, test_loader, conf.model.device)
    preds = []
    for vp in predictions:
        preds.extend(vp)

    p = np.vstack((predictions))
    return p


def test_one(conf: Dict):
    verbose = False
    if "verbose" in conf:
        verbose = bool(conf.verbose)
    datasets_csv = conf.datasets.test.csv
    df = pd.read_csv(datasets_csv,nrows=21)
    test_images = df.path.values.tolist()

    folds = get_test_folds(conf)
    total_preds = []

    use_tta = False
    if "tta" in conf.datasets.test:
        use_tta = bool(conf.datasets.test.tta)
        
    for f in folds:
        if verbose:
            print(f'Testing fold {f} for model size {conf.model.size[0]} x {conf.model.size[1]}')
        preds = []
        model = load_model(conf, f)        
        if use_tta == True:
            if verbose:
                print("TTA Enabled")
            preds = predict_tta(conf, test_images, model)
        else:
            if verbose:
                print("TTA Disabled")
            preds = predict(conf, test_images, model)
        total_preds.append(preds)        

    return total_preds    


def test(config_files):    
    total_predictions = []
    number_predictions = 0
    tta_multiplier = 3
    for config_file in config_files:
        number_config_predictions = config.datasets.train.num_folds
        if "folds" in config_file.datasets.test:
            number_config_predictions = len(config_file.datasets.test.folds)
        if "tta" in config_file.datasets.test and bool(config_file.datasets.test.tta):            
            number_config_predictions=number_config_predictions*tta_multiplier
        preds = test_one(config_file)        
        total_predictions.append(preds)    
        number_predictions+=number_config_predictions
        
    total_predictions = np.vstack(total_predictions)
    # print(total_predictions)
    # print(len(total_predictions))
        
    print(f'Total of {number_predictions} predictions made for the test-set')    
    total_predictions = sum(total_predictions)/len(total_predictions)    
    final_preds = [np.argmax(p) for p in total_predictions]
    print_report(config_files, final_preds)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Path[s] of configuration file[s] needed.")
        exit(1)

    config_files = []
    for arg in sys.argv[1:]:
        print(f'Loading configuration "{arg}"')
        config = Config.load_json(arg)
        print("Configuration")
        pprint(config)

        # config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU
        config.model.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Use 2nd GPU
        
        config_files.append(config)

    print("Evaluate using test dataset.")
    # test(config, config.datasets.test.csv)
    test(config_files)
    # print("Evaluate using real dataset.")
    # test(config, config.datasets.real.csv)
