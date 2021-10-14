#!/bin/bash

#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=4:gpus=1:rhel7
#PBS -l pmem=8gb
#PBS -A sam77_e_g_gc_default


#load python from SW Stack
module load anaconda3

# Go to submission directory
cd $PBS_O_WORKDIR

#load cuda
module load cuda/10.2.89


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/aci/sw/anaconda3/2020.07_gcc-4.8.5-bzb/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/aci/sw/anaconda3/2020.07_gcc-4.8.5-bzb/etc/profile.d/conda.sh" ]; then
        . "/opt/aci/sw/anaconda3/2020.07_gcc-4.8.5-bzb/etc/profile.d/conda.sh"
    else
        export PATH="/opt/aci/sw/anaconda3/2020.07_gcc-4.8.5-bzb/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#Activate environment
conda activate /storage/home/d/dzb5732/work/.dda

python ./ADDA.py $TF

