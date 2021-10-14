#!/usr/bin/env python
# coding: utf-8

# imports 
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
import itertools
from collections import Counter
import gzip

# local imports 
from utils import datasets,samplers,models,utils


# DRCN
class encoder(nn.Module):
    
    def __init__(self, args):
        super(encoder, self).__init__()
        self.featurizer=models.TFCNN(channels=args.feat_size[0], 
                             conv_filters=args.conv_filters, 
                             conv_kernelsize=args.conv_kernelsize, 
                             maxpool_size=args.maxpool_size, 
                             maxpool_strides=args.maxpool_strides)
        
    def forward(self, x_in):
        x_in = self.featurizer(x_in)
        return x_in
    
class decoder(nn.Module):
    
    def __init__(self, args):
        super(decoder, self).__init__()
        self.dconv0 = nn.ConvTranspose1d(args.conv_filters, 120, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm1d(120, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose1d(120, 64, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose1d(64,32, kernel_size=3, stride=2, padding=4)
        self.bn2 = nn.BatchNorm1d(32, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=4)
        self.bn3 = nn.BatchNorm1d(16, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4 = nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2, padding=4)
        self.bn4 = nn.BatchNorm1d(8, affine=False)
        self.prelu4 = nn.PReLU()
        self.dconv5 = nn.ConvTranspose1d(8, 4, kernel_size=2, stride=1, padding=2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.prelu0(self.bn0(self.dconv0(x)))
        x = self.prelu1(self.bn1(self.dconv1(x)))
        x = self.prelu2(self.bn2(self.dconv2(x)))
        x = self.prelu3(self.bn3(self.dconv3(x)))
        x = self.prelu4(self.bn4(self.dconv4(x)))
        x = self.softmax(self.dconv5(x))
        return x

class DRCN(nn.Module):
    
    def __init__(self, args):
        super(DRCN, self).__init__()
        self.featurizer=encoder(args)
        self.classifier=models.TFLSTM(input_features=args.conv_filters, lstm_nodes=args.lstm_outnodes, 
                               fc1_nodes=args.linear1_nodes)
        
        self.decoder=decoder(args)
        
    def forward(self, x_in, apply_sigmoid=False):
        x_in = self.featurizer(x_in)
        class_out = self.classifier(x_in, apply_sigmoid=apply_sigmoid)
        return class_out
    
    def reconstruct(self, x):
        x = self.featurizer(x)
        recon_x = self.decoder(x)
        return recon_x
    
# helper functions for training 
def load_drcn_from_hybrid(drcn, hybrid_path):
    hybrid_state_dict = torch.load(hybrid_path)
    
    pretrained_dict = {}

    for k,v in hybrid_state_dict.items():
        if k.startswith("featurizer"):
            newk = "featurizer." + k
            pretrained_dict[newk] = v
        else:
            pretrained_dict[k] = v
    
    drcn.load_state_dict(pretrained_dict, strict=False)
    
    return drcn


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad=requires_grad
    return


# train function
def train_drcn(args):
    
    # Load the dataset
    logging.debug(f'Loading source and target data...')
    src_dataset, tgt_dataset = datasets.load_data(args)
    
    # Initializing models
    logging.debug(f'Initializing model...')
    classifier = DRCN(args)
    logging.debug(classifier)
    hybrid_path = args.model_state_file.replace(MODEL_NAME, "hybrid")
    if os.path.exists(hybrid_path):
        classifier = load_drcn_from_hybrid(classifier, hybrid_path)
    else:
        logging.error("Hybrid model path not found. Results will not be as expected!")
        
    classifier.to(args.device)
    model_params = utils.get_n_params(classifier)
    logging.debug(f"The model has {model_params} parameters.")
        
    # Defining loss functions, optimizers
    bce_loss_func = nn.BCEWithLogitsLoss()
    mse_loss_func = nn.MSELoss()
    opt = optim.Adam(classifier.parameters(), lr=args.learning_rate, eps=1e-7)
    
    logging.debug("Making samplers...")
    # weighted train and unweighted valid samplers for classifier part of the model
    src_dataset.set_split("train")
    train_sampler = samplers.get_sampler(src_dataset, weighted=True, mini=False)
    nsamples = train_sampler.num_samples
    
    # unweighted sampler of target data for reconstruction loss 
    tgt_dataset.set_split("train")
    tgt_sampler = samplers.get_sampler(tgt_dataset, weighted=False, mini=False)
    
    ##### Training Routine #####
    
    try:
        for epoch_index in range(args.num_epochs):

            # Iterate over training dataset

            # setup: batch generator (w), tgt_batch_generator (uw)
            # set loss and acc to 0, 
            # set train mode on
            src_dataset.set_split('train')
            batch_generator = utils.generate_batches(src_dataset, sampler=train_sampler,
                                               batch_size=args.batch_size, 
                                               device=args.device)

            tgt_dataset.set_split('train')
            tgt_batch_generator = utils.generate_batches(tgt_dataset, sampler=tgt_sampler,
                                               batch_size=args.batch_size, 
                                               device=args.device)
            
            class_running_loss = 0.0
            recon_running_loss = 0.0
            classifier.train()

            for batch_index, (batch_dict, tgt_batch_dict) in enumerate(zip(batch_generator, tgt_batch_generator)):

                if batch_index>500:
                    break

                # the classifier training routine:
                
                # step 1. compute the classifier output for source data
                y_pred = classifier(batch_dict["x_data"].float())

                # step 2. compute the bce loss for classifier output
                loss_class = bce_loss_func(y_pred, batch_dict['y_target'].float())
                loss_class_w = loss_class*1
                
                if batch_index%20==0:

                    # --------------------------------------
                    # zero the gradients
                    opt.zero_grad()

                    # step 3. use optimizer to take gradient step
                    loss_class_w.backward()
                    opt.step()
                
                # -----------------------------------------
                # compute the loss for update
                loss_class_t = loss_class.item()
                class_running_loss += (loss_class_t - class_running_loss) / (batch_index + 1)
                                
                # the reconstructor training routine:
                
                # --------------------------------------
                # zero the gradients
                opt.zero_grad()

                # step 1. compute the reconstructor output for target data
                recon = classifier.reconstruct(tgt_batch_dict["x_data"].float())

                # step 2. compute mse loss of reconstruction
                loss_recon = mse_loss_func(recon, tgt_batch_dict["x_data"].float())
                loss_recon = loss_recon
                loss_recon_w = loss_recon*1
                
                # step 3. use optimizer to take gradient step
                loss_recon_w.backward()
                opt.step()
                
                # -----------------------------------------
                # compute the loss for update
                loss_recon_t = loss_recon.item()
                recon_running_loss += (loss_recon_t - recon_running_loss) / (batch_index + 1)
                
            torch.save(classifier.state_dict(), args.model_state_file)
            logging.debug(f"Reconstruction Loss: {recon_running_loss}, Classification Loss: {class_running_loss}")
            logging.debug(f"Model saved at {args.model_state_file}")

    except KeyboardInterrupt:
        logging.warning("Exiting loop")
    
    return

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
    MODEL_NAME="drcn"
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
        learning_rate=0.0001,
        num_epochs=5,
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
    info = train_drcn(args)

    classifier = DRCN(args)
    # Testing model on source data not required, wasting time can eliminate this step
#     source_dataset = datasets.TFDataset.load_dataset_and_vectorizer_from_path(args.source_csv, 
#                                                                     args.source_genome_fasta, 
#                                                                     ohe=True)
#     utils.eval_model(classifier, source_dataset, args, dataset_type="src", model=MODEL_NAME)

    # Testing model on target dataset
    target_dataset = datasets.TFDataset.load_dataset_and_vectorizer_from_path(args.target_csv, 
                                                                    args.target_genome_fasta, 
                                                                    ohe=True)
    utils.eval_model(classifier, target_dataset, args, dataset_type="tgt", model=MODEL_NAME)
