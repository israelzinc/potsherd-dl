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


from pprint import pprint
from sklearn import metrics
from lib import engine, models
from lib.utils import Dict, Config, create_data_loader, create_model, store_metrics
from lib.es import EarlyStopping



def train(conf: Dict, fold=None):
    
    train_loader, valid_loader, _, valid_targets = create_data_loader(conf, fold)
    model, model_path = create_model(conf, fold)

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

    print('Training....')
    if fold is not None:
        print(f'Fold = {fold}.')
    
    validation_losses = []
    validation_accuracies = []
    training_losses = []
    training_accuracies = []
    for epoch in range(conf.training.epoch):
                
        training_acc, training_loss = engine.train_fn(
            model, train_loader, optimizer, conf.model.device
        )
                

        predictions, valid_loss = engine.evaluate(
            model, valid_loader, conf.model.device
        )
        
        
        print(f'training_loss: {training_loss}')
        print(f'valid_loss: {valid_loss}')
        print(f'training_accuracy: {training_acc}')

        # Unravel batches predictions
        preds = []
        for vp in predictions:
            preds.extend(vp)

        predictions = [torch.argmax(p) for p in preds]
        predictions = np.vstack((predictions)).ravel()

        # acc = metrics.cohen_kappa_score(valid_targets, predictions, weights="quadratic")
        valid_acc = metrics.accuracy_score(valid_targets, predictions)

        # Store the results
        training_accuracies.append(training_acc)
        training_losses.append(training_loss)
        validation_accuracies.append(valid_acc)
        validation_losses.append(valid_loss)

        scheduler.step(valid_acc)
        es(valid_acc, model, model_path)
        if es.early_stop:
            print("Early Stop")
            break

        print(f"Model = {conf.model.type}, Epoch = {epoch}, acc={valid_acc}")
    return (training_accuracies, training_losses,validation_accuracies,validation_losses)


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
    
    config.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use 1st GPU    
    # config.model.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Use 2nd GPU
    # M1/M2 macs
    # config.model.device = torch.device("mps")  # Use M family
    # train(config)
    folds = int(config.datasets.train.num_folds)
    

        
    for f in range(0,5):            
    # for f in range(0,1):            
        fold_metrics = train(config,f)
        store_metrics(config, fold_metrics, f)        
        
