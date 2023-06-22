#!/bin/sh

source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate artfct_env


python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 01.00 2 > ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out
python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 20.00 2 >> ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out
python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 00.20 2 >> ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out

python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 01.00 4 >> ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out
python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 20.00 4 >> ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out
python noise.train_NN.py rap_NN007_single_artifact/clean_percent_050 XV0 00.20 4 >> ~/noise/weights/rap_NN007_single_artifact/clean_percent_050/XV0/train.20201203.out






conda deactivate
conda deactivate


