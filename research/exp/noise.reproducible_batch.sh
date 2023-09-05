#!/bin/sh

source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate artfct_env

counter=1

for tf in {1..10}
do
    for np in {1..10}
    do
        python noise.test_NN.mc_reproducible.py rap_NN007_cw_CLR-003_mc-ommit $counter $tf $np
        counter=$((counter+1))
    done
done

conda deactivate
conda deactivate





