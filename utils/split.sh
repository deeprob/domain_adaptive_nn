#!/bin/bash

set -uex

# this script transfers all.all files in {raw_data}/{genome}/{tf} to their respective {data}/{genome}/{tf} dir

data_dir="../data/"
genomes=("hg38/" "mm10/")
tfs=("CEBPA/" "CTCF/" "Hnf4a/" "RXRA/")


for genome in "${genomes[@]}"
do
    for tf in "${tfs[@]}"
    do
        python make_splits.py -f ${data_dir}${genome}${tf}all.all
        echo "split ${data_dir}${genome}${tf}all.all and saved at ${data_dir}${genome}${tf}split_data.csv.gz"
    done

done
