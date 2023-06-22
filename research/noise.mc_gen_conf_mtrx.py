import time
import pandas as pd
import os,sys
from ast import literal_eval
# from keras.utils import to_categorical
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
np.set_printoptions(suppress=True)

###########################################################
# We want to use the prediction probability to generate an accuracy.
# We can either compute the maximum value to predict one category
# or we can estimate a threshold to use to identify the different categories
###########################################################


def get_clean_prob(st_list):
    st_array = np.array(st_list)
    clean_prob = np.zeros((st_array.shape[0],2))
    clean_prob[:,0] = st_array[:,0]
    clean_idx = st_array.shape[1]//2
    clean_prob[:,1] = st_array[:,clean_idx]
    return clean_prob

def get_metrics(clean_prob_arr,t):
    # column 0 is ground truth and it's either 0 or 1, so we compare to 0.5 
    tp = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] > t))
    fn = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] <= t))
    fp = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] > t))
    tn = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] <= t))

    eps = sys.float_info.epsilon
    acc = 1.0*(tp+tn)/(tp+tn+fp+fn+eps)
    tpr = 1.0*tp/(tp+fn+eps)
    fpr = 1.0 - 1.0*tn/(tn+fp+eps)
    return acc,tpr,fpr,tp,fn,fp,tn

def append_corners(df):
    cols = list(df)
    nb_pos = df.iloc[0].tp + df.iloc[0].fn
    nb_neg = df.iloc[0].fp + df.iloc[0].tn

    r1_values = [1.0,0.0,0.0,0.0,0,nb_pos,0,nb_neg]
    r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
    row1 = pd.DataFrame(r1_dict,index=[0])
    df = row1.append(df,ignore_index=True)

    rN_values = [0.0,0.0,1.0,1.0,nb_pos,0,nb_neg,0]
    rN_dict = {k:v for (k,v) in zip(cols,rN_values)}
    rowN = pd.DataFrame(rN_dict,index=[0])
    df = df.append(rowN,ignore_index=True)

    return df


def gen_df_clean(df,col = 'st1_mean'):
    st_list = [literal_eval(item) for item in list(df[col])]
    clean_prob_arr = get_clean_prob(st_list)
    thresholds=[0.99999,0.9999,0.999]+[0.01*(99-i) for i in range(99)]+[1e-3,1e-4,1e-5]
    df_clean = pd.DataFrame(columns=['thresholds','accuracy','tpr','fpr','tp','fn','fp','tn'])
    df_clean['thresholds'] = thresholds
    for t in df_clean['thresholds']:
        acc,tpr,fpr,tp,fn,fp,tn = get_metrics(clean_prob_arr,t)
        df_clean.loc[df_clean['thresholds']==t, ['accuracy','tpr','fpr','tp','fn','fp','tn']] = [acc,tpr,fpr,tp,fn,fp,tn]
    df_clean = append_corners(df_clean)
    return df_clean


def get_youden(df,col='st1_mean'):
    df_clean = gen_df_clean(df,col)
    df_clean['j'] = df_clean['tpr'] - df_clean['fpr']
    j_max = np.argmax(np.array(df_clean['j']))
    j_max_row = df_clean.iloc[[j_max]]
    th = round(j_max_row['thresholds'].values[0],4)
    return th

def get_lbl_pred_idx(st,th=0.5):
    # threshold
    # th = 0.5
    # lbl is ground truth and it's either 0 or 1, so we compare to 0.5
    lbl = st[:len(st)//2]
    lbl_th = [l>0.5 for l in lbl]
    lbl_index = lbl_th.index(max(lbl_th))

    pred = st[len(st)//2:]
    pred_index = 1
    if pred[0]>th:
        pred_index = 0
    # pred_th = [p>th for p in pred]
    # pred_index = pred_th.index(max(pred_th))

    return lbl_index,pred_index


def compute_var(df,col='st1'):
    col_mean = col + '_mean'
    col_var = col + '_var'
    conf_index = compute_confusion(df,col = col_mean)
    st_list = [literal_eval(item) for item in list(df[col_mean])]
    nb_out = len(st_list[0])//2
    conf_var = np.zeros((nb_out,nb_out))
    th = get_youden(df)
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st,th)
        st_var = literal_eval(list(df.iloc[[index]][col_var])[0])
        conf_var[pred_index,lbl_index] += st_var[pred_index+nb_out]
    eps = sys.float_info.epsilon
    conf_var = np.divide(conf_var,conf_index+eps)
    return conf_var



def compute_mean(df,col='st1'):
    col_mean = col + '_mean'
    conf_index = compute_confusion(df,col = col_mean)
    st_list = [literal_eval(item) for item in list(df[col_mean])]
    nb_out = len(st_list[0])//2
    conf_mean = np.zeros((nb_out,nb_out))
    th = get_youden(df)
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st,th)
        st_mean = literal_eval(list(df.iloc[[index]][col_mean])[0])
        conf_mean[pred_index,lbl_index] += st_mean[pred_index+nb_out]
    eps = sys.float_info.epsilon
    conf_mean = np.divide(conf_mean,conf_index+eps)
    return conf_mean


def compute_confusion(df,col='st1_mean'):
    st_list = [literal_eval(item) for item in list(df[col])]
    nb_out = len(st_list[0])//2
    conf = np.zeros((nb_out,nb_out))
    th = 0.5#get_youden(df)
    print('Threshold is : {}'.format(th))
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st,th)
        conf[pred_index,lbl_index] += 1
    return conf



def plot_conf(conf,fn,outcomes=['clean','artifact'],index_mode=True,title=None):
    plt.figure(figsize=(10,10))
    cmap=plt.cm.Blues
    fnt_size = 25
    if conf.shape[0]<3:
        fnt_size = 35
    # fig=plt.figure(fig_nb,figsize=(5,5))
    imgplot=plt.imshow(conf, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(outcomes))
    cbar = plt.colorbar(imgplot,fraction=0.046, pad=0.04) # ticks=[0,10,20,30,40,50])
    cbar.ax.tick_params(labelsize=fnt_size)
    plt.xticks(tick_marks, outcomes, rotation=45, fontsize=fnt_size)
    plt.yticks(tick_marks, outcomes, fontsize=fnt_size)
    plt.ylabel('Inferred artifact',fontsize=fnt_size+2)
    plt.xlabel('Target artifact',fontsize=fnt_size+2)
    if title:
        plt.title(title,fontsize=fnt_size+2)
    plt.tight_layout()
    # print(conf)
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if (not index_mode) and (conf[i,j]<1.0):
            plt.text(j, i, '{0:0.3f}'.format(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")
        elif conf[i,j]>0:
            plt.text(j, i, int(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")

    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()

def get_2x2(conf):
    conf_2x2 = np.zeros((2,2))
    conf_2x2[0,0] = conf[0,0]
    conf_2x2[0,1] = np.sum(conf[0,1:])
    conf_2x2[1,0] = np.sum(conf[1:,0])
    conf_2x2[1,1] = np.sum(conf[1:,1:])
    tp = conf_2x2[0,0]
    fp = conf_2x2[0,1]
    fn = conf_2x2[1,0]
    tn = conf_2x2[1,1]

    acc = 1.0*(tp+tn)/(tp+tn+fp+fn)
    tpr = 1.0*tp/(tp+fn)
    tnr = 1.0*tn/(tn+fp)
    print(acc,tpr,tnr)
    return conf_2x2


def plot_n_save_conf_mtrx(fig_dir,df):
    ep = os.path.basename(fig_dir)
    outcomes = ['clean','intensity','motion','coverage']
    outcomes_2x2 = ['clean','artifact']
    print(fig_dir)
    if 'nb_classes_02' in fig_dir:
        outcomes = outcomes_2x2

    st1_conf = compute_confusion(df,col='st1_mean')
    print(st1_conf)
    st1_fn = os.path.join(fig_dir,'st1_index_confusion.png')
    plot_conf(st1_conf,st1_fn,outcomes,index_mode=True,title=ep)
    st1_conf_2x2 = get_2x2(st1_conf)
    print(st1_conf_2x2)
    st1_fn = os.path.join(fig_dir,'st1_index_confusion_2x2.png')
    plot_conf(st1_conf_2x2,st1_fn,outcomes_2x2,index_mode=True,title='')

    st1_mean_conf = compute_mean(df,col='st1')
    print(st1_mean_conf)
    st1_fn = os.path.join(fig_dir,'st1_mean_confusion.png')
    plot_conf(st1_mean_conf,st1_fn,outcomes,index_mode=False,title=ep)

    st1_var_conf = compute_var(df,col='st1')
    print(st1_var_conf)
    st1_fn = os.path.join(fig_dir,'st1_var_confusion.png')
    plot_conf(st1_var_conf,st1_fn,outcomes,index_mode=False,title=ep)


    st2_conf = compute_confusion(df,col='st2_mc_mean')
    print(st2_conf)
    st2_fn = os.path.join(fig_dir,'st2_index_confusion.png')
    plot_conf(st2_conf,st2_fn,outcomes)
    st2_conf_2x2 = get_2x2(st2_conf)
    print(st2_conf_2x2)
    st2_fn = os.path.join(fig_dir,'st2_index_confusion_2x2.png')
    plot_conf(st2_conf_2x2,st2_fn,outcomes_2x2,index_mode=True,title='')

    st2_mc_mean_conf = compute_mean(df,col='st2_mc')
    print(st2_mc_mean_conf)
    st2_fn = os.path.join(fig_dir,'st2_mean_confusion.png')
    plot_conf(st2_mc_mean_conf,st2_fn,outcomes,index_mode=False,title=ep)

    st2_var_conf = compute_var(df,col='st2_mc')
    print(st2_var_conf)
    st2_fn = os.path.join(fig_dir,'st2_var_confusion.png')
    plot_conf(st2_var_conf,st2_fn,outcomes,index_mode=False,title=ep)


def print_XVset_target_count(df):
    st1_conf = compute_confusion(df,col='st1_mean')
    print(np.sum(st1_conf,axis=0))

def mc_gen_conf_mtrx_by_epoch(mc_root_dir,fig_dir,sub_experiment,XV_set):
    print(mc_root_dir,sub_experiment)
    epochs_dir = glob.glob(os.path.join(mc_root_dir,sub_experiment))
    print(epochs_dir)
    # epochs_dir = glob.glob(os.path.join(mc_root_dir,'002-ramp_clean_050_to_098_ep0500','epoch_transience','ep*'))
    for ed in sorted(epochs_dir):
        start_time_ed = time.time()
        fn = os.path.join(ed,'{}_stats_summarized.csv'.format(XV_set))
        print('Working on the following file : {}'.format(fn))
        df = pd.read_csv(fn,index_col=0)
        fig_dir_ep = os.path.join(fig_dir,os.path.basename(ed))
        if not os.path.exists(fig_dir_ep):
            os.makedirs(fig_dir_ep,exist_ok=True)
        print_XVset_target_count(df)
        plot_n_save_conf_mtrx(fig_dir_ep,df)
        elapsed_time = time.time() - start_time_ed
        print('Time it took to compute confusion : {0:0.2f} seconds'.format(elapsed_time))


start_time = time.time()

# This script will execute after running noise.mc_summarize_prediction.py
# python noise.mc_gen_conf_mtrx.py rap_NN007_100ep_CLR test nb_classes_02 nb_samples_factor_01.00
XV_set = sys.argv[1]
sub_experiment = sys.argv[2]

experiment = 'rap_NN008_multiple_artifact/clean_percent_098' # sys.argv[1]
XV_nb = 'XV0' # sys.argv[2]
classes = 'nb_classes_02' # sys.argv[4]
factor = 'nb_samples_factor_01.00' # sys.argv[5]

pred_dir = '/trials/data/rpizarro/noise/prediction/'
mc_root_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor)
# mc_root_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor,'epoch_transience')
fig_root_dir = '/trials/data/rpizarro/noise/figs/performance/mc/'
fig_dir = os.path.join(fig_root_dir,experiment,XV_nb,XV_set,classes,factor)
# fig_dir = os.path.join(fig_root_dir,experiment,XV_nb,XV_set,classes,factor,'epoch_transience')

mc_gen_conf_mtrx_by_epoch(mc_root_dir,fig_dir,sub_experiment,XV_set)

elapsed_time = time.time() - start_time
print('Time it took to summarize all epochs : {0:0.2f} seconds'.format(elapsed_time))



