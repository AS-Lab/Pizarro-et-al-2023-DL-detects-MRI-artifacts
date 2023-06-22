#!/bin/sh

# source /data/datasets/shared/rpizarro/miniconda3/bin/activate
conda activate tf_210_env
# conda activate artfct_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# python noise.train_NN.py 50 001-random_constant_clean_098_ep0050_focal_default False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20230206.001-random_constant_clean_ep050_NN8_focal_default.out

# python noise.test_NN.stg1.py 1 001-random_constant_clean_stg1_ep0200_NN7_combat False clean_percent_stg1 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_stg1/XV0/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_stg1_ep0200_NN7_combat/valid_combat.out


# python noise.train_NN.py 200 001-random_constant_clean_stg1_ep0200_NN7_combat False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_stg1/XV0/train.20221201.001-random_constant_clean_ep200_NN7_combat.out

# on hold for now
# python noise.train_NN.py 200 001-random_constant_clean_stg1_ep0200_NN7_domain False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_stg1/XV0/train.20221201.001-random_constant_clean_ep200_NN7_domain.out

# python noise.train_NN.py 200 001-random_constant_clean_stg1_ep0100_NN7_REDO False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_stg1/XV0/train.20221128.001-random_constant_clean_ep100_NN7_REDO.out

#python noise.train_NN.py 60 002-E1ramp_020_data_800_to_20000_ep0060_N2 True 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220819.002-E1ramp_020_ep060_N2.out



# python noise.train_NN.py 50 001-random_constant_clean_098_ep0050_focal_alpha_small False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20221104.001-random_constant_clean_ep050_focal_alpha_small.out

# python noise.train_NN.py 50 001-random_constant_clean_098_ep0050_focal False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220916.001-random_constant_clean_ep050_focal.out




# python noise.test_NN.mc.py 3 001-random_constant_clean_098_ep0050_focal_default False clean_percent_098 test >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal_default/test_mc100.out

# python noise.test_NN.mc.py 3 001-random_constant_clean_098_ep0050_focal_by10 False clean_percent_098 test >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal_by10/test_mc100.out





# python noise.train_NN.py 100 002-E1ramp_100_data_800_to_20000_ep0100_focal True 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220908.002-E1ramp_100_ep100_focal.out


# python noise.train_NN.py 60 002-E1ramp_020_data_800_to_20000_ep0060_N3 True 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220824.002-E1ramp_020_ep060_N3.out


# python noise.train_NN.py 100 002-E1ramp_100_data_800_to_20000_ep0100 True >> ~/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20210811.002-E1ramp_100_factor20_ep100.out
# python noise.test_NN.mc.py 50 002-E1ramp_100_data_800_to_20000_ep0100 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_20.00/002-E1ramp_100_data_800_to_20000_ep0100/valid_mc100.out




# python noise.train_NN.py 53 002-E1ramp_005_data_800_to_20000_ep0053_N3 True >> ~/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20210818.002-E1ramp_005_ep053_N3.out
# python noise.test_NN.mc.py 50 002-E1ramp_005_data_800_to_20000_ep0053_N3 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_005_data_800_to_20000_ep0053_N3/valid_mc100.out

# python noise.train_NN.py 100 002-E1ramp_100_data_800_to_20000_ep0100_N3 True >> ~/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20210824.002-E1ramp_100_ep100_N3.out
# python noise.test_NN.mc.py 50 002-E1ramp_100_data_800_to_20000_ep0100_N3 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_100_data_800_to_20000_ep0100_N3/valid_mc100.out

# python noise.test_NN.mc.py 1 001-initialized_constant_clean_098_ep0050 True valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-initialized_constant_clean_098_ep0050/longitudinal/valid_mc001.out


# python noise.train_NN.py 25 001-initialized_constant_clean_098_ep0050_flip False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220217.001-initialized_ep050_flip.out

python noise.test_NN.aleatoric.py 8 001-initialized_constant_clean_098_ep0050_flip False clean_percent_098 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-initialized_constant_clean_098_ep0050_flip/valid_nonMC.out

python noise.train_NN.py 50 001-initialized_constant_clean_098_ep0050_flip False 10 >> /trials/data/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20220217.001-initialized_ep050_flip.out


# python noise.test_NN.mc.py 50 001-initialized_constant_clean_098_ep0050_N3 False valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-initialized_constant_clean_098_ep0050_N3/valid_mc100.out


# python noise.train_NN.py 250 002-E1ramp_200_data_800_to_20000_ep0250 True 50 >> ~/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/train.20210906.002-E1ramp_200_ep250.out
# python noise.test_NN.mc.py 50 002-E1ramp_200_data_800_to_20000_ep0250 False clean_percent_098 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_200_data_800_to_20000_ep0250/valid_mc100.out

# python noise.test_NN.mc.py 100 002-E1ramp_100_data_800_to_20000_ep0100 False clean_percent_098 test >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_100_data_800_to_20000_ep0100/test_mc100.out

# python noise.test_NN.mc.py 100 001-initialized_constant_clean_098_ep0050_N2 False clean_percent_098 test >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/001-initialized_constant_clean_098_ep0050_N2/test_mc100.out


# python noise.test_NN.mc.py 100 002-E1ramp_200_data_800_to_20000_ep0250 False clean_percent_098 valid >> ../prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_200_data_800_to_20000_ep0250/valid_mc100.out

# python noise.test_NN.mc.py 100 001-CLR-trg12_NN8mod_priming_ep500 False clean_percent_050 test >> ../prediction/rap_NN008_multiple_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_01.00/001-CLR-trg12_NN8mod_priming_ep500/test_mc100.out



conda deactivate


