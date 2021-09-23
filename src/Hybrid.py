#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import sys
import os
import numpy as np
import logging

import argparse
from argparse import Namespace
import tqdm
import itertools
from collections import Counter
import gzip

from utils import datasets,samplers,models,utils


# # Hybrid CNN-RNN
class TFHybrid(nn.Module):
    
    def __init__(self, args):
        super(TFHybrid, self).__init__()
        self.featurizer=models.TFCNN(channels=args.feat_size[0], 
                              conv_filters=args.conv_filters, conv_kernelsize=args.conv_kernelsize, 
                              maxpool_size=args.maxpool_size, maxpool_strides=args.maxpool_strides)
        self.classifier=models.TFLSTM(input_features=args.conv_filters, lstm_nodes=args.lstm_outnodes, 
                               fc1_nodes=args.linear1_nodes)
    
        pass
    
    def forward(self, x_in, apply_sigmoid=False):
        x_in = self.featurizer(x_in)
        x_in = self.classifier(x_in)
        
        if apply_sigmoid:
            x_in = torch.sigmoid(x_in)

        return x_in


def train_hybrid(args):
    
    # Load the dataset
    logging.debug(f'Loading dataset and creating vectorizer...')
    dataset = datasets.TFDataset.load_dataset_and_vectorizer_from_path(args.source_csv, args.source_genome_fasta, ohe=True)    
    
    # Initializing model
    logging.debug(f'Initializing model...')
    classifier = TFHybrid(args)
    classifier = classifier.to(args.device)
    model_params = utils.get_n_params(classifier)
    logging.debug(f"The model has {model_params} parameters.")
        
    # Defining loss function, optimizer and scheduler
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, eps=1e-7)
    # adjusting the learning rate for better performance
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)    
    
    # Making samplers
    train_sampler, valid_sampler = samplers.make_train_samplers(dataset, args)
    logging.debug(f"Training {model_params} parameters with {train_sampler.num_samples} instances at a rate of {round(train_sampler.num_samples/model_params, 6)} instances per parameter.")
    
    # Defining initial train state
    train_state = utils.make_train_state(args)
    
    ##### Training Routine #####
    
    try:
        for epoch_index in range(args.num_epochs):
            logging.debug(f"Starting epoch: {epoch_index}")
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = utils.generate_batches(dataset, sampler=train_sampler,
                                               batch_size=args.batch_size, 
                                               device=args.device)
            running_loss = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):

                # the training routine as follows:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'].float())

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the loss for update
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

            train_state['train_loss'].append(running_loss)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('valid')
            batch_generator = utils.generate_batches(dataset, sampler=valid_sampler,
                                               batch_size=int(args.batch_size*1e1), 
                                               device=args.device)
            running_loss = 0.
            ## TODO::Calculate actual aps
            tmp_filename = f"./{TF}_hybrid_tmp.tmp"
            tmp_file = open(tmp_filename, "wb")
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred = classifier(x_in=batch_dict['x_data'].float())
                y_target = batch_dict['y_target'].float()

                # step 3. compute the loss
                loss = loss_func(y_pred, y_target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # save data for computing aps
                for yp, yt in zip(torch.sigmoid(y_pred).cpu().detach().numpy(), y_target.cpu().detach().numpy()):
                    tmp_file.write(bytes(f"{yp},{yt}\n", "utf-8"))

            train_state['val_loss'].append(running_loss)
            
            # compute aps from saved file
            tmp_file.close()
            val_aps = utils.compute_aps_from_file(tmp_filename)
            os.remove(tmp_filename)
        
            train_state['val_aps'].append(val_aps)

            train_state = utils.update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])
            
            logging.debug(f"Epoch: {epoch_index}, Validation Loss: {running_loss}, Validation APS: {val_aps}")


            if train_state['stop_early']:
                logging.debug("Early stopping criterion fulfilled!")
                break

    except KeyboardInterrupt:
        logging.warning("Exiting loop")
    
    return train_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("tf", help="The Transcription Factor: CTCF/CEBPA/Hnf4a/RXRA")
    parser.add_argument("--sg", help="source genome", default="mm10")
    parser.add_argument("--tg", help="target genome", default="hg38")
    parser.add_argument("--sgf", help="source genome fasta", default="../genomes/mm10_no_alt_analysis_set_ENCODE.fasta")
    parser.add_argument("--tgf", help="target genome fasta", default="../genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    parser.add_argument("-p", help="pilot study", action="store_true")

    args_cli = parser.parse_args()

    TF = args_cli.tf
    SOURCE_GENOME = args_cli.sg
    SOURCE_GENOME_FASTA = args_cli.sgf
    TARGET_GENOME = args_cli.tg
    TARGET_GENOME_FASTA = args_cli.tgf
    PILOT_STUDY=args_cli.p
    MODEL_NAME="hybrid"
    PYTORCH_DEVICE="cuda"
    MODEL_STORAGE_SUFFIX="_pilot" if PILOT_STUDY else ""
    TRAIN=True

    # Logger config
    logging.basicConfig(filename=f'./log/{TF}_{MODEL_NAME}{MODEL_STORAGE_SUFFIX}.log', filemode='w', level=logging.DEBUG)

    args = Namespace(
        # Data and Path information
        model_state_file=f'{MODEL_NAME}{MODEL_STORAGE_SUFFIX}.pth',
        source_csv=f'../data/{SOURCE_GENOME}/{TF}/split_data.csv.gz',
        source_genome_fasta=SOURCE_GENOME_FASTA,
        target_csv = f'../data/{TARGET_GENOME}/{TF}/split_data.csv.gz',
        target_genome_fasta = TARGET_GENOME_FASTA,
        model_save_dir=f'../torch_models/{SOURCE_GENOME}/{TF}/{MODEL_NAME}/',
        results_save_dir=f'../results/{SOURCE_GENOME}/{TF}/',
        feat_size=(4, 500),
        
        # Model hyper parameters
        conv_filters=240,
        conv_kernelsize=20,
        maxpool_strides=15,
        maxpool_size=15,
        lstm_outnodes=32,
        linear1_nodes=1024,
        dropout_prob=0.5,
        
        # Training hyper parameters
        batch_size=128,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=15,
        tolerance=1e-3,
        seed=1337,
        
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True if PYTORCH_DEVICE=="cuda" else False,
        expand_filepaths_to_save_dir=True,
        pilot=PILOT_STUDY, # 2% of original dataset
        train=TRAIN,
        test_batch_size=int(2e3)
    )

    if args.expand_filepaths_to_save_dir:

        args.model_state_file = os.path.join(args.model_save_dir,
                                            args.model_state_file)
        
        logging.info("Expanded filepaths: ")
        logging.info("\t{}".format(args.model_state_file))
        
    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    logging.info("Using CUDA: {}".format(args.cuda))

    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Set seed for reproducibility
    utils.set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    utils.handle_dirs(args.model_save_dir)
    utils.handle_dirs(args.results_save_dir)

    # train model
    train_state = train_hybrid(args)
    # TODO: log train state 

    classifier = TFHybrid(args)
    # Testing model on source dataset
    source_dataset = datasets.TFDataset.load_dataset_and_vectorizer_from_path(args.source_csv, 
                                                                          args.source_genome_fasta, 
                                                                          ohe=True)
    utils.eval_model(classifier, source_dataset, args, dataset_type="src")

    # Testing model on target dataset
    target_dataset = datasets.TFDataset.load_dataset_and_vectorizer_from_path(args.target_csv, 
                                                                 args.target_genome_fasta, 
                                                                 ohe=True)
    utils.eval_model(classifier, target_dataset, args, dataset_type="tgt")
