import pandas as pd
import numpy as np
import os,sys
np.set_printoptions(suppress=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

###########################################################
# We want to use the prediction probability to generate an accuracy.
# We can either compute the maximum value to predict one category
# or we can estimate a threshold to use to identify the different categories
###########################################################


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



def gen_acc_col(df):
    lbl_NN_cols = list(df)[2:-1]
    lbl_cols = lbl_NN_cols[:len(lbl_NN_cols)//2]
    cols = [l.replace('lbl_','') for l in lbl_cols]
    lbl_rename = {a:b for a,b in zip(lbl_cols,cols)}
    lbl_df = df[lbl_cols].rename(columns=lbl_rename)

    NN_cols = lbl_NN_cols[len(lbl_NN_cols)//2:]
    NN_rename = {a:b for a,b in zip(NN_cols,cols)}
    NN_df = df[NN_cols].rename(columns=NN_rename)
    return lbl_df,NN_df,cols


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



def gen_df_clean(thresholds,lbl_df,NN_df,cols):
    df_clean = pd.DataFrame(columns=['thresholds','accuracy','tpr','fpr','tp','fn','fp','tn'])
    df_clean['thresholds'] = thresholds
    for t in df_clean['thresholds']:
        df_t = pd.DataFrame(columns=['lbl_category'])
        df_t['lbl_category'] = lbl_df['clean']>t
        df_t['NN_category'] = NN_df['clean']>t
        acc,tpr,fpr,tp,fn,fp,tn = calculate_metrics(df_t,True)
        df_clean.loc[df_clean['thresholds']==t, ['accuracy','tpr','fpr','tp','fn','fp','tn']] = [acc,tpr,fpr,tp,fn,fp,tn]
        # print('(threshold,accuracy,tpr,tnr) : ({},{},{},{})'.format(t,acc,tpr,fpr))
    df_clean = append_corners(df_clean)
    return df_clean


def plot_roc(save_dir,df,t):
    fig_nb = int(t*10000)
    plt.figure(fig_nb)
    plt.plot(df.fpr,df.tpr)
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    fpr_threshold = float(df[df['thresholds']==t]['fpr'])
    tpr_threshold = float(df[df['thresholds']==t]['tpr'])
    plt.plot(fpr_threshold,tpr_threshold,'r*',markersize=20)
    # plt.tight_layout()
    plt.grid(True)
    fn = os.path.join(save_dir,'clean_t{0:0.5f}.roc.png'.format(t))
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def plot_roc_threshold(save_dir,df_clean):
    save_dir = os.path.join(save_dir,'roc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    for t in df_clean['thresholds']:
        plot_roc(save_dir,df_clean,t)

def plot_conf_threshold(save_dir,df_clean):
    save_dir = os.path.join(save_dir,'conf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    outcomes = ['clean','artifact']
    for index,row in df_clean.iterrows():
        conf = np.zeros((2,2))
        conf[0,0] = row['tp']
        conf[0,1] = row['fp']
        conf[1,0] = row['fn']
        conf[1,1] = row['tn']
        plot_conf(save_dir,conf,outcomes,row['thresholds'])

def plot_conf(save_dir,conf,outcomes,t):
    fig_nb = int(t*10000)
    cmap=plt.cm.Blues
    fnt_size = 13
    if conf.shape[0]<3:
        fnt_size = 20
    fig=plt.figure(fig_nb,figsize=(5,5))
    imgplot=plt.imshow(conf, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(outcomes))
    cbar = plt.colorbar(imgplot,fraction=0.046, pad=0.04) # ticks=[0,10,20,30,40,50])
    cbar.ax.tick_params(labelsize=fnt_size)
    plt.xticks(tick_marks, outcomes, rotation=45, fontsize=fnt_size)
    plt.yticks(tick_marks, outcomes, fontsize=fnt_size)
    plt.ylabel('Inferred artifact',fontsize=fnt_size+2)
    plt.xlabel('Target artifact',fontsize=fnt_size+2)
    plt.tight_layout()
    print(conf)
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i,j]>0:
            plt.text(j, i, int(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")
    fn = os.path.join(save_dir,'clean_t{0:0.5f}.confusion.png'.format(t))
    print('Saving to : {}'.format(fn))
    fig.savefig(fn,bbox_inches='tight')
    plt.close(fig_nb)



# python noise.visual_NN_roc.py rap_NN007_100ep_CLR test
experiment = sys.argv[1]
XV_set = sys.argv[2]

save_dir = '/home/rpizarro/noise/prediction/{}'.format(experiment)
fn = os.path.join(save_dir,'label_prob_{}.csv'.format(XV_set))
df = pd.read_csv(fn,index_col=0)
print('Generating the ROC curve for : {}'.format(fn))

lbl_df,NN_df,cols = gen_acc_col(df)

thresholds=[0.99999,0.9999,0.999]+[0.01*(99-i) for i in range(99)]+[1e-3,1e-4,1e-5]

df_clean = gen_df_clean(thresholds,lbl_df,NN_df,cols)
print(df_clean)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#         print(df_clean)

save_dir = '/home/rpizarro/noise/figs/performance/roc/{}/{}'.format(experiment,XV_set)
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)


plot_conf_threshold(save_dir,df_clean)

plot_roc_threshold(save_dir,df_clean)

