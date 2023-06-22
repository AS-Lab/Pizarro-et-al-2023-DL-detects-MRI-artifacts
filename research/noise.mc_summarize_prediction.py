import time
import pandas as pd
import os,sys
import glob
from ast import literal_eval
from keras.utils import to_categorical
import numpy as np
np.set_printoptions(suppress=True)
_EPS = 1e-5

###########################################################
# We want to use the prediction probability to generate an accuracy.
# We can either compute the maximum value to predict one category
# or we can estimate a threshold to use to identify the different categories
###########################################################


def uncertainty_metrics_classification(prd):
    """
    author: Raghav Meta, Tal Arbel 
    Arguments:
      - prd: post-softmax                                             (mcs, batch_size, num_classes)
             where, mcs: monte carlo samples
                    batch_size: batch_size of deep networks. 
                                basically all samples for which you want to estimate uncertainties
                    num_classes: total classes in your classification problem. 
                                 Keep in mind that for binary classification the functions expects num_classes to be 2. 
    Returns:
      - Entropy: entropy of MC samples across classes                 (batch_size)
      - MI: Mutual Information (bald) of MC samples across classes    (batch_size)
      - sample_variance: Variance of MC samples across classes        (batch_size)
      - mean: Mean of MC samples. DO argmax of this for predictin     (batch_size, num_classes)

    Refer: https://arxiv.org/pdf/1703.02910.pdf and https://arxiv.org/pdf/1808.01200.pdf for more info about Entropy and MI
    """
    entropy = -np.sum(np.mean(prd, 0) * np.log2(np.mean(prd, 0) + _EPS), -1)

    expected_entropy = -np.mean(np.sum(prd * np.log2(prd + _EPS), -1), 0)
    MI = entropy - expected_entropy

    # print(np.var(prd,0))
    sample_variance = np.mean(np.var(prd,0),-1)
    
    mean = np.mean(prd, 0)
    
    return MI, entropy, sample_variance, mean 



def summarize_prediction(df,cols):
    st1_lbl_pred = list(df[cols].mean())
    st1_var = list(df[cols].var())

    lbl_prd = df[cols].to_numpy()
    prd = lbl_prd[:,len(cols)//2:]
    MI, entropy, sample_variance, mean = uncertainty_metrics_classification(prd)

    st1_final_index = compute_index(st1_lbl_pred)
    # print(st1_final_index)

    st2_mc_indices = []
    for index,row in df.iterrows():
        st2_mc_indices.append(compute_index(list(row[cols])))
    # print(mc_index)
    st2_mc_index_mean = list(np.mean(np.array(st2_mc_indices),axis=0))
    st2_mc_index_var = list(np.var(np.array(st2_mc_indices),axis=0))
    st2_final_index = compute_index(st2_mc_index_mean)
    # print(st2_final_index)
    # sys.exit()
    # median = list(df[cols].median())
    # mode = list(df[cols].mode().to_numpy()[0])
    return st1_lbl_pred,st1_var,st1_final_index,MI, entropy, sample_variance,st2_mc_index_mean,st2_mc_index_var,st2_final_index

def compute_index(lbl_pred):
    # print(lbl_pred)
    lbl = lbl_pred[:len(lbl_pred)//2]
    lbl_index = to_categorical(lbl.index(max(lbl)), num_classes=len(lbl))
    # lbl = [lbl[0],sum(lbl[1:])]
    pred = lbl_pred[len(lbl_pred)//2:]
    pred_index = to_categorical(pred.index(max(pred)), num_classes=len(pred))
    # pred = [pred[0],sum(pred[1:])]
    return list(np.concatenate((lbl_index,pred_index),axis=None))

def join_csv_files(mc_pred_dir,XV_set):
    # search_path = os.path.join(mc_pred_dir,'mc*label_prob_{}.csv'.format(XV_set))
    search_path = os.path.join(mc_pred_dir,'mc*label_prob_test.csv')
    print('Looking for file in :{}'.format(search_path))
    files_found = glob.glob(search_path)
    files_found.sort()
    fn0 = files_found[0]
    df = pd.read_csv(fn0,index_col=0)
    if df.shape[1]<5:
        df_000 = pd.read_csv(os.path.join(os.path.dirname(mc_pred_dir),'mc000_label_prob_test.csv'))
        df['path test'] = df_000['path test']
    # df0 has the paths, we did not save in all files to reduce storage size
    df0 = df
    print('Generating the prediction accuracy for training files starting with : {} ... '.format(fn0))
    for fn in files_found[1:]:
        # print('Adding file : {}'.format(os.path.basename(fn)))
        dfi = pd.read_csv(fn,index_col=0)
        df0.update(dfi)
        df = df.append(df0,ignore_index=True)
    return df

def ommit_train_files(train,ommit):
    train_ommit = train.loc[~train['path train'].isin(ommit['path'])].reset_index(drop=True)
    return train_ommit


def mc_summarize_prediction_by_epoch(mc_root_dir,sub_experiment,XV_set):
    epochs_dir = glob.glob(os.path.join(mc_root_dir,sub_experiment))
    # epochs_dir = glob.glob(os.path.join(mc_root_dir,'002-ramp_clean_050_to_098_ep0500','epoch_transience','ep*'))
    for ed in sorted(epochs_dir):
        save_dir = ed
        fn = os.path.join(save_dir,'{}_stats_summarized.csv'.format(XV_set))
        if os.path.exists(fn):
            print('We have already analyzed dir : {}'.format(ed))
            print('Results can be found here : {}'.format(fn))
            continue
        start_time_ed = time.time()
        mc_pred_dir = os.path.join(ed,XV_set)

        print('Working in the following data dir : {}'.format(mc_pred_dir))
        print('Script will write summary to : {}'.format(save_dir))

        df = join_csv_files(mc_pred_dir,XV_set=XV_set)

        cols = list(df)
        path_col = 'path {}'.format('test')
        # path_col = 'path {}'.format(XV_set)
        cols.remove(path_col)
        gk = df.groupby([path_col])

        paths = []

        st1_lbl_pred_all = []
        st1_var_all = []
        st1_final_index_all = []
        MI_all = []
        entropy_all = []
        sample_variance_all = []

        st2_mc_index_mean_all = []
        st2_mc_index_var_all = []
        st2_final_index_all = []

        for name, group in gk: #list(gk)[:20]:
            # print(name)
            paths.append(name)
            st1_lbl_pred,st1_var,st1_final_index,MI, entropy, sample_variance,st2_mc_index_mean,st2_mc_index_var,st2_final_index = summarize_prediction(group,cols)
            
            st1_lbl_pred_all.append(st1_lbl_pred)
            st1_var_all.append(st1_var)
            st1_final_index_all.append(st1_final_index)
            
            MI_all.append(MI)
            entropy_all.append(entropy)
            sample_variance_all.append(sample_variance)

            st2_mc_index_mean_all.append(st2_mc_index_mean)
            st2_mc_index_var_all.append(st2_mc_index_var)
            st2_final_index_all.append(st2_final_index)

            if st1_final_index != st2_final_index:
                print(name)
                print('Mismatch in st1 and st2:')
                print('st1 : {}'.format(st1_final_index))
                print('st2 : {}'.format(st2_final_index))

            # print(mean_lbl_pred,var)

        d = {'path':paths, 'st1_mean':st1_lbl_pred_all, 'st1_var':st1_var_all, 'st1_index':st1_final_index_all, 
            'MI':MI_all,'entropy':entropy_all,'sample_variance':sample_variance_all,
            'st2_mc_mean':st2_mc_index_mean_all,'st2_mc_var':st2_mc_index_var_all, 'st2_mc_index':st2_final_index_all}
        df = pd.DataFrame(d)
        print('Writing summarized predictions to : {}'.format(fn))
        df.to_csv(fn)

        elapsed_time = time.time() - start_time_ed
        print('Time it took to summarize mc inferences : {0:0.2f} seconds'.format(elapsed_time))


start_time = time.time()

# python noise.mc_summarize_prediction.py rap_NN007_100ep_CLR XV3 test nb_classes_02 nb_samples_factor_01.00

clean_percent = sys.argv[1]
XV_set = sys.argv[2]
sub_experiment = sys.argv[3]

experiment = 'rap_NN008_multiple_artifact/{}'.format(clean_percent)
XV_nb = 'XV0' # sys.argv[2]
classes = 'nb_classes_02' # sys.argv[4]
factor = 'nb_samples_factor_01.00' # sys.argv[5]

pred_dir = '/trials/data/rpizarro/noise/prediction/'
mc_root_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor)
# mc_root_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor,'epoch_transience')


# mc_pred_dir = '/home/rpizarro/noise/prediction/{}/{}/{}/{}/{}'.format(experiment,XV_nb,classes,factor,XV_set)

mc_summarize_prediction_by_epoch(mc_root_dir,sub_experiment,XV_set)

elapsed_time = time.time() - start_time
print('Time it took to summarize all epochs : {0:0.2f} seconds'.format(elapsed_time))

