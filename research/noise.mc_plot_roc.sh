#!/bin/sh

source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate artfct_env


python noise.mc_plot_roc.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_samples_factor_01.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_samples_factor_01.00/noise.mc_plot_roc.out
python noise.mc_plot_roc.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_samples_factor_05.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_samples_factor_05.00/noise.mc_plot_roc.out
python noise.mc_plot_roc.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_samples_factor_20.00 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_samples_factor_20.00/noise.mc_plot_roc.out
python noise.mc_plot_roc.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_samples_factor_00.20 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_samples_factor_00.20/noise.mc_plot_roc.out
python noise.mc_plot_roc.py rap_NN007_single_artifact/clean_percent_050 XV0 valid nb_samples_factor_00.05 > ../prediction/rap_NN007_single_artifact/clean_percent_050/XV0/nb_samples_factor_00.05/noise.mc_plot_roc.out





conda deactivate
conda deactivate


