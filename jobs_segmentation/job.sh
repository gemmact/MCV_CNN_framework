#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 16000 # 2GB solicitados.
#SBATCH -p mhigh,mlow # or mlow Partition to submit to
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o ../logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e ../logs/%x_%u_%j.err # File to which STDERR will be written

python ../main.py --exp_name SemSeg_fcn8_Camvid --exp_folder ../experiments/ --config_file ../config/segmentation/SemSeg_sample_fcn8_Camvid.yml


