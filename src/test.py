#!/usr/bin/env python3

import sys
import torch
import pandas as pd
import numpy as np
import albumentations as A

from pprint import pprint
from sklearn.metrics import classification_report
from os.path import join
from lib import engine, models
from lib.dataset import Dataset
from lib.utils import Dict, Config, print_confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_model(conf: Dict):
    model = models.select(conf.model.type, conf.model.n_classes)
    model_path = join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{config.datasets.train.val_fold}.bin")
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else 'cpu')
    )
    model.to(conf.model.device)

    return model


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


def test(conf: Dict, datasets_csv: str):
    df = pd.read_csv(datasets_csv)
    test_images = df.path.values.tolist()

    model = load_model(conf)
    preds = predict(conf, test_images, model)
    final_preds = [np.argmax(p) for p in preds]

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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Path[s] of configuration file[s] needed.")
        exit(1)

    for arg in sys.argv[1:]:
        print(f'Loading configuration "{arg}"')
        config = Config.load_json(arg)
        print("Configuration")
        pprint(config)

        # config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU
        config.model.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Use 2nd GPU

        print("Evaluate using testing dataset.")
        test(config, config.datasets.test.csv)
        print("Evaluate using real dataset.")
        test(config, config.datasets.real.csv)
