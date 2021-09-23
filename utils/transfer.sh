#!/bin/bash

set -uex

# this script transfers all.all files in ../../cross-species-domain-adaptation/{raw_data}/{genome}/{tf} to the ../{data}/{genome}/{tf} dir

input_dir="../../cross-species-domain-adaptation/raw_data/"
output_dir="../data/"
genomes=("hg38/" "mm10/")
tfs=("CEBPA/" "CTCF/" "Hnf4a/" "RXRA/")


for genome in "${genomes[@]}"
do
    for tf in "${tfs[@]}"
    do
        cp ${input_dir}${genome}${tf}all.all ${output_dir}${genome}${tf}all.all
        echo "copied ${input_dir}${genome}${tf}all.all to ${output_dir}${genome}${tf}all.all"
    done

done
