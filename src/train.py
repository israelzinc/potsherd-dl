#!/usr/bin/env python3
###############################################################################
# This script is the command to train the models.
#
# Configuration files found at ./conf
#
# PARAMETERS:
# argv[1] is the configuration file. Just use one of the provided ones
#
# RETURN VALUE:
# This script does not return any value, but the models are stored in the models folder
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import sys
import torch
import numpy as np
import pandas as pd
import albumentations as A

from pprint import pprint
from os.path import join
from sklearn import metrics
from lib import engine, models
from lib.utils import Dict, Config
from lib.dataset import Dataset
from lib.es import EarlyStopping


def get_paths_labels(csv_path: Dict, val_fold: int) -> tuple:
    df = pd.read_csv(csv_path)

    df_train = df[df.kfold != val_fold].reset_index(drop=True)
    df_val = df[df.kfold == val_fold].reset_index(drop=True)

    # Return (1) path of train images, (2) path of val images, (3) train label, (4) val labels
    return df_train.path.values, df_val.path.values, df_train.class_num.values, df_val.class_num.values


def create_data_loader(conf: Dict) -> tuple:
    train_imgs, val_imgs, train_targets, val_targets = get_paths_labels(
        config.datasets.train.csv,
        config.datasets.train.val_fold
    )

    train_aug = A.Compose(
        [
            A.SmallestMaxSize(conf.model.size),
            A.CenterCrop(conf.model.size[0], conf.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]
    )

    val_aug = A.Compose(
        [
            A.SmallestMaxSize(conf.model.size),
            A.CenterCrop(conf.model.size[0], conf.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_imgs, train_targets, augmentations=train_aug, channel_first=True),
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset(val_imgs, val_targets, augmentations=val_aug, channel_first=True),
        batch_size=conf.DataLoader.batch_size,
        num_workers=conf.DataLoader.num_workers,
        shuffle=None
    )

    return train_loader, val_loader, train_targets, val_targets


def create_model(conf: Dict) -> tuple:
    model = models.select(conf.model.type, conf.model.n_classes)
    model = model.to(device=conf.model.device)

    model_path = join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{config.datasets.train.val_fold}.bin")

    return model, model_path


def train(conf: Dict):
    train_loader, valid_loader, _, valid_targets = create_data_loader(conf)
    model, model_path = create_model(conf)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.optimizer.learning_rate,
        betas=conf.optimizer.betas,
        eps=conf.optimizer.epsilon,
        weight_decay=conf.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=conf.scheduler.patience,
        mode=conf.scheduler.mode
    )
    es = EarlyStopping(
        patience=conf.scheduler.patience,
        mode=conf.scheduler.mode
    )

    print('training....')
    for epoch in range(conf.training.epoch):

        training_loss = engine.train_fn(
            model, train_loader, optimizer, conf.model.device
        )

        predictions, valid_loss = engine.evaluate(
            model, valid_loader, conf.model.device
        )
        print(f'training_loss: {training_loss}')
        print(f'valid_loss: {valid_loss}')

        # Unravel batches predictions
        preds = []
        for vp in predictions:
            preds.extend(vp)

        predictions = [torch.argmax(p) for p in preds]
        predictions = np.vstack((predictions)).ravel()

        # acc = metrics.cohen_kappa_score(valid_targets, predictions, weights="quadratic")
        acc = metrics.accuracy_score(valid_targets, predictions)

        scheduler.step(acc)
        es(acc, model, model_path)
        if es.early_stop:
            print("Early Stop")
            break

        print(f"Model = {conf.model.type}, Epoch = {epoch}, acc={acc}")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("\nError: Path[s] of configuration file[s] needed.")
        print("\nUsage: ./train.py <configuration_file>\n")        
        exit(1)

    config = sys.argv[1]
    
    print(f'Loading configuration "{config}"')
    config = Config.load_json(config)
    print("Configuration")
    pprint(config)    
    
    config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU
    # config.model.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Use 2nd GPU
    train(config)
