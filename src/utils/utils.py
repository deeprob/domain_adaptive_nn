#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import average_precision_score
import gzip
import tqdm
import gzip
from .samplers import get_test_sampler


def generate_batches(dataset, sampler, batch_size, shuffle=False,
                    drop_last=True, device="cpu"):
    
    # define the dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                           sampler=sampler, shuffle=shuffle,
                           drop_last=drop_last, num_workers=4,
                           pin_memory=False)
    
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items(): # TODO: Look at datadict.keys(), no need for tensor here
            if name!= "genome_loc":
                out_data_dict[name] = data_dict[name].to(device)
            else:
                out_data_dict[name] = data_dict[name]
        yield out_data_dict


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 0,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_aps': [],
            'val_loss': [],
            'val_aps': [],
            'test_loss': -1,
            'test_aps': -1,
            'model_filename': args.model_state_file}


def make_train_state_adda(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 0,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_aps': [],
            'val_loss': [],
            'val_aps': [],
            'save_model': False}


def make_train_state_ae(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e9,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'val_loss': []}


def update_train_state_adda(args, train_state):
    """Handle the training state updates. Determines whether to stop model training early

    Components:
     - Early Stopping: Prevent overfitting.

    :param args: main arguments
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save model if performance improved
    if train_state['epoch_index'] >= 1:
        apc_tm1, apc_t = train_state['val_aps'][-2:] # looking at the last two validation aps

        # If apc worsened or remained almost the same within a tolerance level
        if apc_t <= train_state['early_stopping_best_val'] or np.abs(apc_tm1 - apc_t)<args.tolerance:
            train_state['save_model'] = False
            # Update step
            train_state['early_stopping_step'] += 1 # updating early stopping info
        # apc increased
        else:
            # Save the best model
            train_state['early_stopping_best_val'] = apc_t
            train_state['save_model'] = True
                
            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def update_train_state(args, model, train_state):
    """Handle the training state updates. Determines whether to stop model training early

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        curr_aps = train_state['val_aps'][0]
        train_state['early_stopping_best_val'] = curr_aps
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        apc_tm1, apc_t = train_state['val_aps'][-2:] # looking at the last two validation aps

        # If apc worsened 
        if apc_t <= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1 # updating early stopping info
        # apc increased
        else:
            # Save the best model
            if apc_t > train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = apc_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def update_train_state_ae(train_state):
    """Handle the training state updates. Determines whether to stop model training early

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        apc_tm1, apc_t = train_state['val_loss'][-2:] # looking at the last two validation aps

        # If apc worsened or remained almost the same within a tolerance level
        if apc_t >= train_state['early_stopping_best_val'] or np.abs(apc_tm1 - apc_t)<args.tolerance:
            # Update step
            train_state['early_stopping_step'] += 1 # updating early stopping info
        # apc increased
        else:
            # Save the best model
            if apc_t < train_state['early_stopping_best_val']:
                train_state['early_stopping_best_val'] = apc_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def compute_aps(y_pred, y_target):
    """Computes the average precision score"""
    y_target = y_target.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return average_precision_score(y_target, y_pred)

def compute_aps_from_file(file):
    """Computes the aps score from a file"""
    results_df = pd.read_csv(file, header=None)
    y_target = results_df[1].values
    y_pred = results_df[0].values
    return average_precision_score(y_target, y_pred)

def get_split_ratio(bd):
    values = bd["y_target"].cpu().numpy()
    counts = Counter(values)
    return counts[1]/counts[0]


def get_n_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return


def save_test_pred(filename, y_preds, y_targets, genomic_locs, mode="ab"):
    y_preds = y_preds.cpu().detach().numpy()
    y_targets = y_targets.cpu().detach().numpy()
    genomic_locs = list(map(lambda x: x.numpy() if type(x)==torch.Tensor else x, genomic_locs))
    
    with gzip.open(filename, mode) as f:
        for y_pred, y_target, chrm, start, end in zip(y_preds, y_targets, genomic_locs[0], genomic_locs[1], genomic_locs[2]):
            f.write(bytes(f"{y_pred},{y_target},{chrm},{start},{end}\n", "utf-8"))
    return


def eval_model(classifier, dataset, args, dataset_split="test", dataset_type="src", model="hybrid", suffix=""):
    """
    classifier initialized before
    dataset of type TFDataset
    """
    
    # Initializing
    classifier.load_state_dict(torch.load(args.model_state_file))
    classifier = classifier.to(args.device)
    loss_func = nn.BCEWithLogitsLoss()
    dataset.set_split(dataset_split)
    
    test_sampler = get_test_sampler(dataset, mini=args.pilot)

    batch_generator = generate_batches(dataset, sampler=test_sampler, shuffle=False, 
                                       batch_size=args.test_batch_size, 
                                       device=args.device, drop_last=False)

    running_loss = 0.
    running_aps = 0.
    y_preds = []
    y_targets = []
    classifier.eval()
    mode = "wb"
    suffix = "_" + suffix if suffix else suffix
    save_file_replace = f"_{dataset_type}{suffix}.csv.gz"
    save_filename = os.path.basename(args.model_state_file).replace(".pth", save_file_replace)
    save_file = os.path.join(args.results_save_dir, save_filename)
    
    # Runnning evaluation routine
    test_bar = tqdm.tqdm(desc=f'split={dataset_split}',
                          total=len(dataset)//args.test_batch_size, 
                          position=0, 
                          leave=True)

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        if model=="hybrid" or model=="drcn":
            y_pred = classifier(x_in=batch_dict['x_data'].float())
        elif model=="dann":
            y_pred, _ = classifier(x_in=batch_dict['x_data'].float())
        save_test_pred(save_file, 
                       torch.sigmoid(y_pred), 
                       batch_dict['y_target'], 
                       batch_dict["genome_loc"], 
                       mode=mode)
        mode = "ab" 

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the average precision score
        aps_t = compute_aps(y_pred, batch_dict['y_target'])
        running_aps += (aps_t - running_aps) / (batch_index + 1)

        # update test bar
        test_bar.set_postfix(loss=running_loss, 
                              aps=running_aps, 
                              batch=batch_index)
        test_bar.update()
    
    return save_file