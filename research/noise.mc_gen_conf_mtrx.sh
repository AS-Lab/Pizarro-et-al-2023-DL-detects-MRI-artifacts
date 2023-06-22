#!/bin/sh

source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate artfct_env


python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_01.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_01.00/noise.mc_gen_conf_mtrx.out
python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_01.00_ep0200 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_01.00_ep0200/noise.mc_gen_conf_mtrx.out
python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_05.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_05.00/noise.mc_gen_conf_mtrx.out
python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_20.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_20.00/noise.mc_gen_conf_mtrx.out
python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_00.20 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_00.20/noise.mc_gen_conf_mtrx.out
python noise.mc_gen_conf_mtrx.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_classes_02 nb_samples_factor_00.05 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_00.05/noise.mc_gen_conf_mtrx.out




conda deactivate
conda deactivate


