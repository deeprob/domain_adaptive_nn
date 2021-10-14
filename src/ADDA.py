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
import copy

# local imports 
from utils import datasets,samplers,models,utils


# ADDA no new models required to be defined
    
# helper functions for training 
def load_encoder(encoder, hybrid_model_path):
    hybrid_model_dict = torch.load(hybrid_model_path)
    enc_dict = {k.replace("featurizer.", "", 1):v for k,v in hybrid_model_dict.items() if k.startswith("featurizer")}
    encoder.load_state_dict(enc_dict)
    return encoder


def load_classifier(classifier, hybrid_model_path):
    hybrid_model_dict = torch.load(hybrid_model_path)
    class_dict = {k.replace("classifier.", "", 1):v for k,v in hybrid_model_dict.items() if k.startswith("classifier")}
    classifier.load_state_dict(class_dict)
    return classifier 


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad=requires_grad
    return

# adda separate evaluation function
def eval_model(encoder, classifier, dataset, args, 
               dataset_split="test", mini=False,  
               save=True, save_file_suffix="", dataset_type="src"):
    
    # initilize models
    classifier = classifier.to(args.device)
    encoder = encoder.to(args.device) 
    
    # initializing loss
    loss_func = nn.BCEWithLogitsLoss()
    
    # samplers and batches
    dataset.set_split(dataset_split)
    
    test_sampler = utils.get_test_sampler(dataset, mini=mini)
    
    batch_generator = utils.generate_batches(dataset, sampler=test_sampler, shuffle=False, 
                                       batch_size=args.test_batch_size, 
                                       device=args.device, drop_last=False)

    
    running_loss = 0.
    running_aps = 0.
    y_preds = []
    y_targets = []
    encoder.eval()
    classifier.eval()
    
    if save:
        mode = "wb"
        save_file_replace = f"_{dataset_type}{save_file_suffix}.csv.gz"
        save_filename = os.path.basename(args.model_state_file).replace(".pth", save_file_replace)
        save_file = os.path.join(args.results_save_dir, save_filename)
        
    if mini:
        nsamples = test_sampler.num_samples
    else:
        nsamples = len(dataset)
    
    # Runnning evaluation routine
    
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(encoder(x_in=batch_dict['x_data'].float()))
        
        if save:
            utils.save_test_pred(save_file, torch.sigmoid(y_pred), batch_dict['y_target'], batch_dict["genome_loc"], mode=mode)
            mode = "ab" 

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the average precision score
        aps_t = utils.compute_aps(y_pred, batch_dict['y_target'])
        running_aps += (aps_t - running_aps) / (batch_index + 1)
    
    return save_file if save else (running_loss, running_aps)


# train function
def train_tgt(src_encoder, tgt_encoder, classifier, discriminator, src_dataset, tgt_dataset, args):

    # samplers for source and target data
    src_dataset.set_split("train")
    src_sampler = samplers.get_sampler(src_dataset, weighted=False, mini=True)
    tgt_dataset.set_split("train")
    tgt_sampler = samplers.get_sampler(tgt_dataset, weighted=False, mini=True)
        
    # encoders, discriminator and classifier initialization
    src_encoder = src_encoder.to(args.device)
    tgt_encoder = tgt_encoder.to(args.device)
    discriminator = discriminator.to(args.device)
    classifier = classifier.to(args.device)
    
    encoder_filename = os.path.join(args.model_save_dir, f"{MODEL_NAME}_tgt_enc.pth")
    discriminator_filename = os.path.join(args.model_save_dir, f"{MODEL_NAME}_dscm.pth")
    
    # Initializing loss, optimizer and scheduler
    loss_func = nn.BCEWithLogitsLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), 
                               lr=0.0001,
                               betas=(0.5, 0.9))
    optimizer_dscm = optim.Adam(discriminator.parameters(), 
                               lr=0.0001,
                               betas=(0.5, 0.9))
    
    ### Training Routine ###
    num_epochs = args.num_epochs
    batch_size=args.batch_size
    
    logging.debug(f"Source samples: {src_sampler.num_samples}, Target samples: {tgt_sampler.num_samples}")
    
    init_src_encoder_dict = copy.deepcopy(src_encoder.state_dict())
    
    try:
        for epoch_index in range(num_epochs):
            
            # verify before every epoch that the src encoder did not change... 
            for t1, t2 in zip(init_src_encoder_dict.values(), src_encoder.state_dict().values()):
                assert torch.equal(t1, t2)

            
            src_dataset.set_split('train')
            src_batch_generator = utils.generate_batches(src_dataset, sampler=src_sampler,
                                               batch_size=batch_size, 
                                               device=args.device)
            tgt_dataset.set_split('train')
            tgt_batch_generator = utils.generate_batches(tgt_dataset, sampler=tgt_sampler,
                                               batch_size=batch_size, 
                                               device=args.device)
            running_loss_dscm = 0.0
            running_loss_tgt = 0.0
            running_domainacc = 0.0
            tgt_encoder.train()
            discriminator.train()

            for batch_index, (src_batch_dict, tgt_batch_dict) in enumerate(zip(src_batch_generator, tgt_batch_generator)):
                
                ### Discriminator Training Routine ###
                if batch_index%10==0:
                    
                    set_requires_grad(src_encoder, requires_grad=False)
                    set_requires_grad(tgt_encoder, requires_grad=False)
                    set_requires_grad(discriminator, requires_grad=True)
                
                    # --------------------------------------
                        # zero the gradients
                    optimizer_dscm.zero_grad()

                    # extract source and target features
                    feat_src = src_encoder(src_batch_dict['x_data'].float())
                    feat_tgt = tgt_encoder(tgt_batch_dict['x_data'].float())
                    feat_concat = torch.cat((feat_src, feat_tgt), 0).detach()
                    assert feat_concat.requires_grad == False

                    # predict
                    pred_concat = discriminator(feat_concat)

                    # prepare labels
                    label_src = torch.ones(batch_size, dtype=torch.float, device=args.device)
                    label_tgt = torch.zeros(batch_size, dtype=torch.float, device=args.device)
                    label_concat = torch.cat((label_src, label_tgt), 0)

                    # compute loss due to source
                    loss_dscm = loss_func(pred_concat, label_concat)

                    # use loss to produce gradients
                    loss_dscm.backward()                

                    # optimizer step
                    optimizer_dscm.step()


                    # compute overall loss
                    loss_t = loss_dscm.item()
                    running_loss_dscm += (loss_t - running_loss_dscm) / (batch_index + 1)


                    # compute domain accuracy
                    domain_hat = torch.sigmoid(pred_concat)>0.5
                    domain_hat = domain_hat.long()
                    acc_domain = torch.sum(domain_hat==label_concat)/len(label_concat)
                    acc_domain = acc_domain.item()
                    running_domainacc += (acc_domain - running_domainacc) / (batch_index + 1)
                
                
                # -----------------------------------------               
                ### Target Encoder Training Routine ###
                
                set_requires_grad(tgt_encoder, requires_grad=True)
                set_requires_grad(discriminator, requires_grad=False)
                
                # Step 1. zero the gradients
                optimizer_dscm.zero_grad()
                optimizer_tgt.zero_grad()
                
                # Step 2. Extract target features
                feat_tgt = tgt_encoder(tgt_batch_dict['x_data'].float())
                assert feat_tgt.requires_grad == True
                                
                # Step 3. Predict using discriminator
                pred_tgt = discriminator(feat_tgt)
                                
                # Step 4. Prepare fake labels
                label_tgt = torch.ones(batch_size, dtype=torch.float, device=args.device)
                
                # Step 5. Compute loss for target encoder
                loss_tgt = loss_func(pred_tgt, label_tgt)
                
                # Step 6. Use loss to produce gradients
                loss_tgt.backward()
                
                # Step 7. optimize target encoder
                optimizer_tgt.step()
                                
                loss_t = loss_tgt.item()
                running_loss_tgt += (loss_t - running_loss_tgt) / (batch_index + 1)
                                
            logging.debug(f"DA: {running_domainacc}, Tgt Enc Loss: {running_loss_tgt}, DSCM Loss: {running_loss_dscm}")

            torch.save(tgt_encoder.state_dict(), encoder_filename)
            torch.save(discriminator.state_dict(), discriminator_filename)
            
                
    except KeyboardInterrupt:       
        logging.info("Exiting loop")
        
    
    return encoder_filename, discriminator_filename


def train_adda(args):
    
    # load dataset
    src_dataset, tgt_dataset = datasets.load_data(args)
    
    
    # initialize models
    src_encoder = models.TFCNN(channels=args.feat_size[0],
                               conv_filters=args.conv_filters, 
                               conv_kernelsize=args.conv_kernelsize,
                               maxpool_size=args.maxpool_size,
                               maxpool_strides=args.maxpool_strides)
    
    src_classifier = models.TFLSTM(input_features=args.conv_filters, 
                            lstm_nodes=args.lstm_outnodes, 
                            fc1_nodes=args.linear1_nodes)
    
    tgt_encoder =models.TFCNN(channels=args.feat_size[0],
                              conv_filters=args.conv_filters, 
                              conv_kernelsize=args.conv_kernelsize,
                              maxpool_size=args.maxpool_size, 
                              maxpool_strides=args.maxpool_strides)
    
    linear_layer_in = int(np.floor((args.feat_size[1] - args.maxpool_size - 2)/args.maxpool_strides + 1)*args.conv_filters)
    discriminator = models.TFMLP(input_features=linear_layer_in, 
                          fc1_nodes=args.linear1_nodes, 
                          dropout_prob=0.5)
    
    
    # load source model from pretrained hybrid model
    logging.debug("==== Loading model for source domain ====")
    logging.debug(">>> Source Encoder <<<")
    logging.debug(f"{src_encoder}")
    logging.debug(">>> Source Classifier <<<")
    logging.debug(f"{src_classifier}")
    
    ## TODO: Load state dict from hybrid model
    hybrid_model_path = args.model_state_file.replace(MODEL_NAME, "hybrid")    
    
    src_encoder = load_encoder(src_encoder, hybrid_model_path) 
    src_classifier = load_classifier(src_classifier,  hybrid_model_path)
    
    src_encoder_filename = os.path.join(args.model_save_dir, 
                                   f"{MODEL_NAME}_src_enc.pth")
    classifier_filename = os.path.join(args.model_save_dir, 
                                   f"{MODEL_NAME}_class.pth")
    
    # save source encoder and classifier
    torch.save(src_encoder.state_dict(), src_encoder_filename)
    torch.save(src_classifier.state_dict(), classifier_filename)    
    
    
    
    # train target encoder by GAN
    logging.debug("==== Training encoder for target domain ====")
    logging.debug(">>> Target Encoder <<<")
    logging.debug(f"{tgt_encoder}")
    logging.debug(">>> Discriminator <<<")
    logging.debug(f"{discriminator}")
    
    # initialize target encoder from source encoder model path
    tgt_encoder.load_state_dict(torch.load(src_encoder_filename))
    # initialize classifier from path
    src_classifier.load_state_dict(torch.load(classifier_filename))
    
    # train target encoder
    tgt_encoder_filename, discriminator_filename = train_tgt(src_encoder, tgt_encoder, 
                                                             src_classifier, discriminator, 
                                                             src_dataset, tgt_dataset, args)

    return src_encoder_filename, classifier_filename, tgt_encoder_filename, discriminator_filename


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
    MODEL_NAME="adda" 
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
        batch_size=512,
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
    src_encoder_filename, classifier_filename, tgt_encoder_filename, discriminator_filename = train_adda(args)

    # evaluate model
    src_encoder = models.TFCNN(channels=args.feat_size[0], 
                               conv_filters=args.conv_filters, 
                               conv_kernelsize=args.conv_kernelsize, 
                               maxpool_size=args.maxpool_size, 
                               maxpool_strides=args.maxpool_strides)
    
    src_classifier = models.TFLSTM(input_features=args.conv_filters, 
                                   lstm_nodes=args.lstm_outnodes, 
                                   fc1_nodes=args.linear1_nodes)
    
    tgt_encoder = models.TFCNN(channels=args.feat_size[0], 
                               conv_filters=args.conv_filters, 
                               conv_kernelsize=args.conv_kernelsize, 
                               maxpool_size=args.maxpool_size, 
                               maxpool_strides=args.maxpool_strides)

   
    tgt_encoder.load_state_dict(torch.load(tgt_encoder_filename))
    src_classifier.load_state_dict(torch.load(classifier_filename))
    
    src_data, tgt_data = datasets.load_data(args)
    
    eval_model(tgt_encoder, src_classifier, tgt_data, args, 
               save_file_suffix="", dataset_type="tgt")