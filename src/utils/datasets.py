#!/usr/bin/env python
# coding: utf-8


from pyfaidx import Fasta
import pandas as pd
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GenomeVocabulary(object):
    """A class to store the original genome as a pyfaidx Fasta object to recover nucleotide sequences from genomic coordinates"""
    
    def __init__(self, genome_fasta):
        """
        genome_fasta: A pyfaidx.fasta object as the genome
        """
        self.genome = genome_fasta
        
    def get_sequence(self, chrom_name, start, end):
        """A method that returns the dna sequence given their chromosomal coordinates"""
        seq = self.genome.get_seq(chrom_name, start, end)
        try:
            assert len(seq) == end-start
        # weird pyfaidx sequence length error
        except AssertionError:
            req_end = end-start
            seq = seq[0:req_end]
        return seq
    
    @classmethod
    def load_from_path(cls, genome_filepath):
        genome_fasta = Fasta(genome_filepath, as_raw=True)
        return cls(genome_fasta)


class GenomeVectorizer(GenomeVocabulary):
    """A class that converts the chromosomal coordinates to numerical encodings"""
    def __init__(self, genome_fasta, ohe=True, k=5):
        """
        genome_fasta: A pyfaidx.fasta object as the genome
        ohe: True if OneHotEncoded sequence is desired
        k: kmer length of ohe is not true
        """
        super(GenomeVectorizer, self).__init__(genome_fasta)
        # dictionary for ohe vectorizer
        self._ohe_dict = {"A":[1, 0, 0, 0],
                         "C":[0, 1, 0, 0],
                         "G":[0, 0, 1, 0],
                         "T":[0, 0, 0, 1],
                         "a":[1, 0, 0, 0],
                         "c":[0, 1, 0, 0],
                         "g":[0, 0, 1, 0],
                         "t":[0, 0, 0, 1]}
        
        self.ohe = ohe
        
        if not self.ohe:
            # dictionary for kmer vectorizer
            self.nt = "ATGC"
            self.k = k
            self._kmer2idx_dict = {"".join(k):i for i, k in enumerate(itertools.product(self.nt, repeat=self.k))}
            self._idx2kmer_dict = {v:k for k,v in self._kmer2idx_dict.items()}
        
        self.vectorize = self._ohe_vectorize if self.ohe else self._kmer_vectorize
        
        
                
    def _ohe_vectorize(self, chrm, start, end):
        """Vectorizer that produces one hot encoded sequence"""
        seq = self.get_sequence(chrm, start, end)
        ohe = np.array([self._ohe_dict.get(nt, [0, 0, 0, 0]) for nt in seq], dtype=np.float32)
        return np.transpose(ohe) # change from 500*4 to 4*500
    
    def _kmer_vectorize(self, chrm, start, end):
        """Vectorizer that produces normalized term frequencies of the kmer present in sequence"""
        seq = self.get_sequence(chrm, start, end)
        
        nt_vocab_len = len(self._kmer2idx_dict)
        arr = np.zeros(nt_vocab_len + 1) # 1 added to accomodate unknowns; last index of array is for unknowns
        
        for i in range(0, len(seq)- self.k + 1):
            # if kmer in dict, count increases, else unknown count increases
            arr[self._kmer2idx_dict.get(seq[i:i+self.k], nt_vocab_len)] += 1
        
        narr = arr/sum(arr)
        return narr[:-1]
    
    @classmethod
    def load_from_path(cls, genome_filepath, ohe=True, k=3):
        genome_fasta = Fasta(genome_filepath, as_raw=True)
        return cls(genome_fasta, ohe=ohe, k=k)
    
    
class TFDataset(Dataset):
    """A Class that contains all genomic coordinates with their labels"""
    
    def __init__(self, tf_df, vectorizer):
        """
        tf_df: A dataframe with 5 columns, chrm, start, end, label, split
        vectorizer: The object that vectorizes this dataset going from genomic coordinates to sequence to numerical encodings
        """
        
        self.tf_df = tf_df
        self._vectorizer = vectorizer
        
        self.set_split("train")
        
        pass
    
    @classmethod
    def load_dataset_and_vectorizer_from_path(cls, tf_df_path, genome_path, nrows=None, ohe=True, k=5):
        """
        tf_df_path: path to the tf csv file with genomic locations and annotations  
        genome_path: path to the genome fasta file of the organism 
        """
        tf_df = pd.read_csv(tf_df_path, nrows=nrows)
        vectorizer = GenomeVectorizer.load_from_path(genome_path, ohe=ohe, k=k)
        return cls(tf_df, vectorizer)
    
    def set_split(self, split="train"):
        self._target_split = split
        self._target_df = self.tf_df if self._target_split=="all" else self.tf_df[self.tf_df.split==split] 
        self._target_size = len(self._target_df)
        return
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        
        row = self._target_df.iloc[index]
        
        chrm, start, end = row.chrm, row.start, row.end
        
        tf_vector = self._vectorizer.vectorize(row.chrm, row.start, row.end)
        
        tf_label = row.label
        
        return {"x_data": tf_vector,
                "y_target": tf_label,
                "genome_loc": (chrm, start, end)}
    
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    
    def get_feature_size(self):
        """
        Func to output the length of the feature vector created by the vectorizer
        NB: This function assumes that all genomic coordinates length in the df 
        is same as the length of the first genomic coordinate length. 
        TODO: Modify to include a check across the dataframe.
        """
        # length of the dna sequence is end - start
        seq_length = self._target_df.iloc[0, 2] - self._target_df.iloc[0, 1]
        # length of the vector is len(seq) * 4 [ohe encoding]
        feat_shape = (4, seq_length) if self._vectorizer.ohe else (len(self._vectorizer._kmer2idx_dict),)
        return feat_shape


def load_data(args):
    source_dataset = TFDataset.load_dataset_and_vectorizer_from_path(args.source_csv, 
                                                              args.source_genome_fasta, 
                                                              ohe=True)
    
    target_dataset = TFDataset.load_dataset_and_vectorizer_from_path(args.target_csv, 
                                                              args.target_genome_fasta, 
                                                              ohe=True)
    return source_dataset, target_dataset