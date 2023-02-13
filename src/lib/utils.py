import json
import numpy as np
import pandas as pd
import torch
from lib.dataset import Dataset
from lib import models
import albumentations as A
import os
from typing import List, Tuple
from PIL import Image

DEFAULT_METRICS_FOLDER = "../metrics/"
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


def print_confusion_matrix(y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_labels = np.unique(y)
    y_pred_labels = np.unique(y_pred)

    matrix = pd.DataFrame(np.zeros((len(y_labels), len(y_pred_labels))),
                          index=y_labels, columns=y_pred_labels, dtype=int)

    for c, p in zip(y, y_pred):
        matrix.loc[c, p] += 1

    return matrix

def get_paths_labels(csv_path: str, val_fold: int) -> tuple:
    df = pd.read_csv(csv_path)

    df_train = df[df.kfold != val_fold].reset_index(drop=True)
    df_val = df[df.kfold == val_fold].reset_index(drop=True)

    # Return (1) path of train images, (2) path of val images, (3) train label, (4) val labels
    return df_train.path.values, df_val.path.values, df_train.class_num.values, df_val.class_num.values

def create_data_loader(config: Dict, fold=None) -> tuple:

    if not fold:
        fold = config.datasets.train.val_fold
    
    # train_imgs, val_imgs, train_targets, val_targets = get_paths_labels(
    #     config.datasets.train.csv,
    #     config.datasets.train.val_fold
    # )

    train_imgs, val_imgs, train_targets, val_targets = get_paths_labels(
        config.datasets.train.csv,
        fold
    )

    train_aug = A.Compose(
        [
            A.SmallestMaxSize(config.model.size),
            A.CenterCrop(config.model.size[0], config.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]
    )

    val_aug = A.Compose(
        [
            A.SmallestMaxSize(config.model.size),
            A.CenterCrop(config.model.size[0], config.model.size[1]),
            A.augmentations.transforms.Normalize()
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_imgs, train_targets, augmentations=train_aug, channel_first=True),
        batch_size=config.DataLoader.batch_size,
        num_workers=config.DataLoader.num_workers,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        Dataset(val_imgs, val_targets, augmentations=val_aug, channel_first=True),
        batch_size=config.DataLoader.batch_size,
        num_workers=config.DataLoader.num_workers,
        shuffle=None
    )

    return train_loader, val_loader, train_targets, val_targets

def create_model(config: Dict, fold=None) -> tuple:
    if fold is None:
        fold = config.datasets.train.val_fold
    model = models.select(config.model.type, config.model.n_classes)
    model = model.to(device=config.model.device)

    # model_path = os.path.join(config.model.dir,
    #                   f"{config.model.id}_{config.model.type}_{config.model.size[0]}_{config.model.size[1]}_{config.datasets.train.val_fold}.bin")
    model_path = os.path.join(config.model.dir,
                      f"{config.model.id}_{config.model.type}_{config.model.size[0]}_{config.model.size[1]}_{fold}.bin")

    return model, model_path

def load_model(conf: Dict, fold=None):
    if fold is None:
        fold = conf.datasets.train.val_fold
    model = models.select(conf.model.type, conf.model.n_classes)
    model_path = os.path.join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{fold}.bin")
        
    # model.load_state_dict(torch.load(
    #     model_path,
    #     map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else "cpu")
    # )
    model.load_state_dict(torch.load(
        model_path,
        map_location=conf.model.device)
    )
    model.to(conf.model.device)

    return model

def get_test_folds(config):
    if "folds" in config.datasets.test:
        return config.datasets.test.folds
    
    return range(0,config.datasets.train.num_folds)

def get_loader_with_augs(config: Dict, augmentations_list: A.transforms ,test_images: List[str]) -> torch.utils.data.DataLoader:
    test_targets = np.zeros(len(test_images), dtype=np.int_)
    augs = A.Compose(
            augmentations_list
        ) 
    test_dataset = Dataset(
        image_paths=test_images,
        targets=test_targets,        
        augmentations=augs,
        channel_first = True,
        torgb=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.DataLoader.batch_size,
        num_workers=config.DataLoader.num_workers,
        shuffle=False        
    )
    return test_loader

def store_metrics(config: Dict, metrics: Tuple[List, List], fold: int):
    out_base = DEFAULT_METRICS_FOLDER
    if "metrics" in config:
        if "dir" in config.metrics:
            out_base = config.metrics.dir

    output_path = os.path.join(out_base,
        f"{config.model.id}_{config.model.type}_{config.model.size[0]}_{config.model.size[1]}_{fold}.csv")

    losses, accuracies = metrics
    df = pd.DataFrame({"loss":losses,"acc":accuracies})

    df.to_csv(output_path)

def saitama(model, conf, image_path):
    IMAGE_SIZE = conf.model.size[0]
    device = conf.model.device  
    image = Image.open(image_path)
    image = image.convert("RGB")
    # orig_image = image

    # Set up the transformations
    # transform_ = transforms.Compose([
    #         transforms.Resize(IMAGE_SIZE),
    #         transforms.CenterCrop(IMAGE_SIZE),            
    #         transforms.ToTensor(),
    # ])

    image = np.array(image)
    test_aug = A.Compose(
            [
                A.SmallestMaxSize(conf.model.size),
                A.CenterCrop(conf.model.size[0], conf.model.size[1]),
                A.augmentations.transforms.Normalize()
            ]
        )
    augmented = test_aug(image=image)
    image = augmented["image"]
    orig_image = image
    # # Transforms the image
    # image = transform_(image)

    # Reshape the image (because the model use 
    # 4-dimensional tensor (batch_size, channel, width, height))
    # image = image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    

    # Set the device for the image
    image = torch.tensor(image)
    image = image.to(device)
    
    image = image.unsqueeze(0)
    

    # Set the requires_grad_ to the image for retrieving gradients
    image.requires_grad_()

    # Retrieve output from the image
    output, _ = model(image)
    
    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    print("Output index",output_idx)

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(IMAGE_SIZE, IMAGE_SIZE)

    # Reshape the image
    # image = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    return (orig_image, saliency, output_idx.cpu())