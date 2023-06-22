import pandas as pd
import numpy as np
import os,sys
from ast import literal_eval
np.set_printoptions(suppress=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

def get_mc_col(df,col='mc_pred_match'):
    mc_pred_match_list = [literal_eval(item) for item in list(df[col])]
    return np.array(mc_pred_match_list)

def get_accs(df):
    mc_pred_match = get_mc_col(df,'mc_pred_match') 
    mc_accuracies = mc_pred_match.mean(axis=0)
    ensemble_accuracy = df['mc_ensemble_acc'].mean()
    return mc_accuracies,ensemble_accuracy

def get_sens_spec(df):
    mc_pred_match_sens = get_mc_col(df[df['lbl_clean']==1.0],'mc_pred_match')
    mc_sens = mc_pred_match_sens.mean(axis=0)
    mc_pred_match_spec = get_mc_col(df[df['lbl_clean']==0.0],'mc_pred_match')
    mc_spec = mc_pred_match_spec.mean(axis=0)
    ensemble_sens = df[df['lbl_clean']==1.0]['mc_ensemble_acc'].mean()
    ensemble_spec = df[df['lbl_clean']==0.0]['mc_ensemble_acc'].mean()
    return mc_sens,ensemble_sens,mc_spec,ensemble_spec


def calculate_tp_fn_fp_tn(df,c):
    tp = ( (df['lbl_category']==c)
            & (df['NN_category']==c) ).sum()
    fn = ( (df['lbl_category']==c)
            & (df['NN_category']!=c) ).sum()
    fp = ( (df['lbl_category']!=c)
            & (df['NN_category']==c) ).sum()
    tn = ( (df['lbl_category']!=c)
            & (df['NN_category']!=c) ).sum()
    return tp,fn,fp,tn




def calculate_metrics(df,c):
    tp,fn,fp,tn = calculate_tp_fn_fp_tn(df,c)
    # print('tp : {}'.format(tp))
    # print('fn : {}'.format(fn))
    # print('fp : {}'.format(fp))
    # print('tn : {}'.format(tn))
    eps = sys.float_info.epsilon
    acc = 1.0*(tp+tn)/(tp+tn+fp+fn+eps)
    tpr = 1.0*tp/(tp+fn+eps)
    fpr = 1.0 - 1.0*tn/(tn+fp+eps)
    return acc,tpr,fpr,tp,fn,fp,tn


def append_cs_corners(df):
    cols = list(df)

    r1_values = [1.0,0.995479,0.0159267,0.852958,0.13863,0.98350,0.0467432,0.246243,0.102758]
    r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
    row1 = pd.DataFrame(r1_dict,index=[0])
    df = row1.append(df,ignore_index=True)

    rN_values = [0.0,0.955966,0.0510948,0.0172489,0.0494974,0.473381,0.150727,0.000823409,0.00281058]
    rN_dict = {k:v for (k,v) in zip(cols,rN_values)}
    rowN = pd.DataFrame(rN_dict,index=[0])
    df = df.append(rowN,ignore_index=True)

    return df


def append_corners(df):
    cols = list(df)

    r1_values = [1.0,0.0,0.0,0.0,0.0,6340.0,0.0,124.0]
    r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
    row1 = pd.DataFrame(r1_dict,index=[0])
    df = row1.append(df,ignore_index=True)

    rN_values = [0.0,0.98004,1.0,1.0,6340.0,0.0,124.0,0.0]
    rN_dict = {k:v for (k,v) in zip(cols,rN_values)}
    rowN = pd.DataFrame(rN_dict,index=[0])
    df = df.append(rowN,ignore_index=True)

    return df

def calculate_conf_stats(t,lbl_df,NN_df):
    lbl_NN_df = NN_df
    lbl_NN_df = lbl_NN_df.rename(columns={'clean':'NN_clean'})
    lbl_NN_df['lbl_clean'] = lbl_df['clean']
    df_tp = lbl_NN_df.loc[ ( lbl_NN_df['lbl_clean']>t ) 
            & ( lbl_NN_df['NN_clean']>t )]
    df_fn = lbl_NN_df.loc[ ( lbl_NN_df['lbl_clean']>t ) 
            & ( lbl_NN_df['NN_clean']<=t )]
    df_fp = lbl_NN_df.loc[ ( lbl_NN_df['lbl_clean']<=t ) 
            & ( lbl_NN_df['NN_clean']>t )]
    df_tn = lbl_NN_df.loc[ ( lbl_NN_df['lbl_clean']<=t ) 
            & ( lbl_NN_df['NN_clean']<=t )]
    tpm = df_tp['clean_mean'].mean()
    tps = df_tp['clean_std'].mean()
    fnm = df_fn['clean_mean'].mean()
    fns = df_fn['clean_std'].mean()
    fpm = df_fp['clean_mean'].mean()
    fps = df_fp['clean_std'].mean()
    tnm = df_tn['clean_mean'].mean()
    tns = df_tn['clean_std'].mean()
    return [tpm,tps,fnm,fns,fpm,fps,tnm,tns]



def gen_df_clean(thresholds,lbl_df,NN_df):
    df_conf_stats = pd.DataFrame(columns=['thresholds','tpm','tps','fnm','fns','fpm','fps','tnm','tns'])
    df_conf_stats['thresholds'] = thresholds

    df_clean = pd.DataFrame(columns=['thresholds','accuracy','tpr','fpr','tp','fn','fp','tn'])
    df_clean['thresholds'] = thresholds
    for t in df_clean['thresholds']:
        df_t = pd.DataFrame(columns=['lbl_category'])
        df_t['lbl_category'] = lbl_df['clean']>t
        df_t['NN_category'] = NN_df['clean']>t
        conf_stats = calculate_conf_stats(t,lbl_df,NN_df)
        df_conf_stats.loc[df_conf_stats['thresholds']==t, ['tpm','tps','fnm','fns','fpm','fps','tnm','tns']] = conf_stats
        acc,tpr,fpr,tp,fn,fp,tn = calculate_metrics(df_t,True)
        df_clean.loc[df_clean['thresholds']==t, ['accuracy','tpr','fpr','tp','fn','fp','tn']] = [acc,tpr,fpr,tp,fn,fp,tn]
        # print('(threshold,accuracy,tpr,tnr) : ({},{},{},{})'.format(t,acc,tpr,fpr))
    df_clean = append_corners(df_clean)
    df_conf_stats = append_cs_corners(df_conf_stats)

    return df_clean,df_conf_stats


def plot_conf(conf,title):
    outcomes = ['clean','artifact']    
    cmap=plt.cm.Blues
    fnt_size = 13
    if conf.shape[0]<3:
        fnt_size = 20
    # fig=plt.figure(fig_nb,figsize=(5,5))
    imgplot=plt.imshow(conf, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(outcomes))
    cbar = plt.colorbar(imgplot,fraction=0.046, pad=0.04) # ticks=[0,10,20,30,40,50])
    cbar.ax.tick_params(labelsize=fnt_size)
    plt.xticks(tick_marks, outcomes, rotation=45, fontsize=fnt_size)
    plt.yticks(tick_marks, outcomes, fontsize=fnt_size)
    plt.ylabel('Inferred artifact',fontsize=fnt_size+2)
    plt.xlabel('Target artifact',fontsize=fnt_size+2)
    plt.title(title,fontsize=fnt_size+2)
    plt.tight_layout()
    print(conf)
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i,j]<1.0:
            plt.text(j, i, '{0:0.3f}'.format(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")
        elif conf[i,j]>0:
            plt.text(j, i, int(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")



def plot_roc_conf(save_dir,df,conf,conf1,conf2,t):
    fig_nb = int(t*10000)
    plt.figure(fig_nb,figsize=(20,15))

    plt.subplot(221)
    plt.plot(df.fpr_mean,df.tpr_mean)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    # error = np.sqrt( df['tpr_std']**2 + df['fpr_std']**2 )
    # plt.fill_between(df.fpr_mean, df.tpr_mean-df.tpr_std, df.tpr_mean+df.tpr_std, interpolate=True,alpha=0.1)
    plt.fill_betweenx(df.tpr_mean, df.fpr_mean-df.fpr_std, df.fpr_mean+df.fpr_std, interpolate=True,alpha=0.1)

    fpr_threshold = float(df[df['thresholds']==t]['fpr_mean'])
    tpr_threshold = float(df[df['thresholds']==t]['tpr_mean'])
    plt.plot(fpr_threshold,tpr_threshold,'r*',markersize=20)
    plt.text(0.6,0.4,'t={0:0.3f}'.format(t),fontsize=17)
    plt.tick_params(labelsize=15)
    plt.xlabel('false positive rate',fontsize=17)
    plt.ylabel('true positive rate',fontsize=17)
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(222)
    plot_conf(conf,'Confusion')
    plt.subplot(223)
    plot_conf(conf1,'Probability')
    plt.subplot(224)
    plot_conf(conf2,'Std. Deviation')

    fn = os.path.join(save_dir,'clean_t{0:0.5f}.roc_conf.png'.format(t))
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)

def get_conf(row,key='_mean'):
    conf = np.zeros((2,2))
    conf[0,0] = row['tp{}'.format(key)]
    conf[0,1] = row['fp{}'.format(key)]
    conf[1,0] = row['fn{}'.format(key)]
    conf[1,1] = row['tn{}'.format(key)]
    return conf


def plot_roc_conf_threshold(save_dir,df,df1):
    save_dir = os.path.join(save_dir,'roc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    for t in df['thresholds']:
        row = df[df['thresholds']==t]
        conf = get_conf(row,'_mean')
        row = df1[df1['thresholds']==t]
        conf_mean = get_conf(row,'_probability_mean')
        row = df1[df1['thresholds']==t]
        conf_std = get_conf(row,'_probability_std')
        plot_roc_conf(save_dir,df,conf,conf_mean,conf_std,t)



def plot_mc_roc_conf(save_dir,df):
    mc_pred_clean = get_mc_col(df,'mc_pred_clean')
    df['mc_pred_clean_std'] = mc_pred_clean.std(axis=1)
    # df1=df[['path','mc_pred_clean_std']]
    # df1 = df1.sort_values(by=['mc_pred_clean_std'])
    lbl_df = pd.DataFrame(columns=['clean'])
    lbl_df['clean'] = df['lbl_clean']
    NN_df = pd.DataFrame(columns=['clean'])
    df_tpr_fpr = pd.DataFrame(columns=['thresholds','tpr_mean','tpr_std','fpr_mean','fpr_std'])
    df_MC_conf_stats = pd.DataFrame(columns=['thresholds'])
    thresholds=[0.99999,0.9999,0.999]+[0.01*(99-i) for i in range(99)]+[1e-3,1e-4,1e-5]
    mc_tpr = []
    mc_fpr = []
    mc_tp = []
    mc_fn = []
    mc_fp = []
    mc_tn = []
    mc_tpm = []
    mc_tps = []
    mc_fnm = []
    mc_fns = []
    mc_fpm = []
    mc_fps = []
    mc_tnm = []
    mc_tns = []
    
    NN_df['clean_mean'] = mc_pred_clean.mean(axis=1)
    NN_df['clean_std'] = mc_pred_clean.std(axis=1)

    for mc in range(5):#  range(mc_pred_clean.shape[1]):
        NN_df['clean'] = mc_pred_clean[:,mc]
        df_clean,df_conf_stats = gen_df_clean(thresholds,lbl_df,NN_df)

        mc_tpr.append(list(df_clean['tpr']))
        mc_fpr.append(list(df_clean['fpr']))
        mc_tp.append(list(df_clean['tp']))
        mc_fn.append(list(df_clean['fn']))
        mc_fp.append(list(df_clean['fp']))
        mc_tn.append(list(df_clean['tn']))

        mc_tpm.append(list(df_conf_stats['tpm']))
        mc_fnm.append(list(df_conf_stats['fnm']))
        mc_fpm.append(list(df_conf_stats['fpm']))
        mc_tnm.append(list(df_conf_stats['tnm']))

        mc_tps.append(list(df_conf_stats['tps']))
        mc_fns.append(list(df_conf_stats['fns']))
        mc_fps.append(list(df_conf_stats['fps']))
        mc_tns.append(list(df_conf_stats['tns']))

    mc_tpr = np.transpose(np.array(mc_tpr))
    mc_fpr = np.transpose(np.array(mc_fpr))
    mc_tp = np.transpose(np.array(mc_tp))
    mc_fn = np.transpose(np.array(mc_fn))
    mc_fp = np.transpose(np.array(mc_fp))
    mc_tn = np.transpose(np.array(mc_tn))

    mc_tpm = np.transpose(np.array(mc_tpm))
    mc_fnm = np.transpose(np.array(mc_fnm))
    mc_fpm = np.transpose(np.array(mc_fpm))
    mc_tnm = np.transpose(np.array(mc_tnm))

    mc_tps = np.transpose(np.array(mc_tps))
    mc_fns = np.transpose(np.array(mc_fns))
    mc_fps = np.transpose(np.array(mc_fps))
    mc_tns = np.transpose(np.array(mc_tns))

    df_tpr_fpr['thresholds'] = df_clean['thresholds']
    df_tpr_fpr['tpr_mean'] = mc_tpr.mean(axis=1)
    df_tpr_fpr['tpr_std'] = mc_tpr.std(axis=1)
    df_tpr_fpr['fpr_mean'] = mc_fpr.mean(axis=1)
    df_tpr_fpr['fpr_std'] = mc_fpr.std(axis=1)
    df_tpr_fpr['error'] = np.sqrt( df_tpr_fpr['tpr_std']**2 + df_tpr_fpr['fpr_std']**2 )
    df_tpr_fpr['tp_mean'] = mc_tp.mean(axis=1)
    df_tpr_fpr['fn_mean'] = mc_fn.mean(axis=1)
    df_tpr_fpr['fp_mean'] = mc_fp.mean(axis=1)
    df_tpr_fpr['tn_mean'] = mc_tn.mean(axis=1)
    print(df_tpr_fpr)

    df_MC_conf_stats['thresholds'] = df_clean['thresholds']
    df_MC_conf_stats['tp_probability_mean'] = mc_tpm.mean(axis=1)
    df_MC_conf_stats['tp_probability_std'] = mc_tps.mean(axis=1)
    df_MC_conf_stats['fn_probability_mean'] = mc_fnm.mean(axis=1)
    df_MC_conf_stats['fn_probability_std'] = mc_fns.mean(axis=1)
    df_MC_conf_stats['fp_probability_mean'] = mc_fpm.mean(axis=1)
    df_MC_conf_stats['fp_probability_std'] = mc_fps.mean(axis=1)
    df_MC_conf_stats['tn_probability_mean'] = mc_tnm.mean(axis=1)
    df_MC_conf_stats['tn_probability_std'] = mc_tns.mean(axis=1)
    print(df_MC_conf_stats)

    plot_roc_conf_threshold(save_dir,df_tpr_fpr,df_MC_conf_stats)



def plot_accuracy(save_dir,df):
    mc_accuracies,ensemble_accuracy = get_accs(df)
    plt.figure(figsize=(5,5))
    plt.grid(True)
    plt.hist(mc_accuracies,color='r')
    plt.axvline(x=ensemble_accuracy,color='b')
    plt.xlabel('accuracy')
    fn = os.path.join(save_dir,'prediction_accuracy.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn,bbox_inches='tight')
    plt.close


def plot_sens_spec(save_dir,df):
    mc_sens,ensemble_sens,mc_spec,ensemble_spec = get_sens_spec(df)
    plt.figure(figsize=(5,10))
    plt.subplot(2,1,1)
    plt.grid(True)
    plt.hist(mc_sens,color='r')
    plt.axvline(x=ensemble_sens,color='b')
    plt.xlabel('sensitivity')
    plt.xlim(0.6,1.0)

    plt.subplot(2,1,2)
    plt.grid(True)
    plt.hist(mc_spec,color='r')
    plt.axvline(x=ensemble_spec,color='b')
    plt.xlabel('specificity')
    plt.xlim(0.6,1.0)
    fn = os.path.join(save_dir,'prediction_sens_spec.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn,bbox_inches='tight')
    plt.close


# python noise.mc_plot_performance.py rap_NN007_100ep_CLR test
experiment = sys.argv[1]
XV_set = sys.argv[2]

pred_dir = '/home/rpizarro/noise/prediction/{}'.format(experiment)
fn = os.path.join(pred_dir,'{}_summarized_prediction.csv'.format(XV_set))
df = pd.read_csv(fn,index_col=0)

save_dir = '/home/rpizarro/noise/figs/performance/mc/{}/{}'.format(experiment,XV_set)
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)

plot_mc_roc_conf(save_dir,df)

sys.exit()

plot_accuracy(save_dir,df)

plot_sens_spec(save_dir,df)





