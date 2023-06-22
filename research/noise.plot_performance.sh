#!/bin/sh

source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate artfct_env


python noise.plot_performance.py rap_NN007_cw_CLR-003_mc XV0
python noise.plot_performance.py rap_NN007_cw_CLR-003_mc XV1
python noise.plot_performance.py rap_NN007_cw_CLR-003_mc XV2
python noise.plot_performance.py rap_NN007_cw_CLR-003_mc XV3
python noise.plot_performance.py rap_NN007_cw_CLR-003_mc XV4

python noise.plot_performance.py rap_NN007_single_artifact XV0
python noise.plot_performance.py rap_NN007_single_artifact XV1
python noise.plot_performance.py rap_NN007_single_artifact XV2
python noise.plot_performance.py rap_NN007_single_artifact XV3
python noise.plot_performance.py rap_NN007_single_artifact XV4





conda deactivate
conda deactivate


