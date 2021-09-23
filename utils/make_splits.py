#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# set numpy seed for validation data reproducibility
seed = 1337
np.random.seed(seed)


class SplitAssigner(object):
    """Assigns train/valid/test split to a tf binding dataframe"""
    
    def __init__(self, df, split_func=None):
        """
        df: A dataframe with 4 columns, chrm (chromosome name), start (genome coord start), end (genome coord end), label (genome location binding 0/1 label)
        split_func: A function which will be used to split the dataframe into train/valid/test data. If None, class methods will be used 
        """
        
        self.df = df
        self.split_func = self._assign_split if not split_func else split_func
        
#         if not split_func:
#             self._generate_valid_index()
        
        self.df["split"] = self.df.apply(self.split_func, axis=1)
        
        pass
    
    @classmethod
    def load_from_path(cls, df_path, sep="\t", nrows=None, split_func=None):
        df = pd.read_csv(df_path, usecols=[0, 1, 2, 3], sep="\t", header=None, nrows=nrows, engine="c")
        df.columns= ["chrm", "start", "end", "label"]
        return cls(df, split_func)
      
    @classmethod
    def load_from_path_and_save_in_chunks(cls, df_path, save_path = None, sep="\t", chunksize=1e6, split_func=None):
        df = pd.read_csv(df_path, usecols=[0, 1, 2, 3], sep="\t", header=None, iterator=True, chunksize=chunksize)
        header=True
        mode="w"
        save_path = save_path if save_path else os.path.join(os.path.dirname(df_path), "split_data.csv.gz")
        
        tqdm_bar = tqdm(desc="chunk progress")
        
        for i,chunk in enumerate(df):
            chunk.columns= ["chrm", "start", "end", "label"]
            split_obj = cls(chunk, split_func)
            split_obj._save_to_disk_chunks(save_path, mode=mode, header=header)

            mode="a"
            header=False

            tqdm_bar.update()
        
        return save_path
    
    
    def _save_to_disk_chunks(self, path, mode, header):
        self.df.to_csv(path, mode=mode, header=header, compression="gzip", index=False)
        return path
    
    def save_to_disk(self, path):
        self.df.to_csv(path, index=False)
        return path
    
    def _assign_split2(self, row):
        # if the chromosome number is 1 or 2, assign test
        if row["chrm"] == "chr1" or row["chrm"] == "chr2":
            return "test"
        # if the row index belongs to validation index set, assign valid 
        elif row.name in self.valid_set:
            return "valid"
        return "train"
    
    def _assign_split(self, row):
        # if the chromosome number is 1, assign valid
        if row["chrm"] == "chr1":
            return "valid"
        # if the chromosome number is 2, assign test 
        elif row["chrm"] == "chr2":
            return "test"        
        return "train"
    
    def _generate_valid_index(self):
        all_idx = self.df[~self.df.chrm.isin(["chr1", "chr2"])].index
        # make sure that df has no duplicate index
        assert np.all(all_idx.duplicated()==False) == True
        # convert to numpy 
        all_idx = np.array(all_idx)
        # draw random samples from the array to assign as validation index
        self.valid_set = set(np.random.choice(all_idx, int(0.1*len(all_idx))))
        return 


# In[20]:


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # transcription factor binding filepath
    parser.add_argument("tf_path", help="TF binding annotation file; bed like format expected", type=str)
    
    # optional output dataframe storing path
    parser.add_argument("-o", "--out", help="output dataframe for storing annotated file with splits", type=str, default=None)
    # optional chucksize argument
    parser.add_argument("-c", "--cs", help="chunksize of the dataframe to iterate over", type=int, default=int(1e6))
    # force to make dataframe
    parser.add_argument("-f", "--force", help="force to create dataframe and overwrite current dataframe", action="store_true")
    
    args = parser.parse_args()
    
    split_path = os.path.join(os.path.dirname(args.tf_path), "split_data.csv.gz") if not args.out else args.out
    if not args.force:
        if os.path.isfile(split_path):
            print("File already exists")
    else:
        # create the file from tf file
        split_path = SplitAssigner.load_from_path_and_save_in_chunks(args.tf_path, save_path=args.out, chunksize=args.cs)      
