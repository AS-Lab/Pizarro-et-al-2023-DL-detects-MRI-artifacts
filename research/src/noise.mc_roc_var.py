import time
import pandas as pd
import os,sys
import glob
from ast import literal_eval
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

def join_csv_files(mc_pred_dir,XV_set):
    search_path = os.path.join(mc_pred_dir,'mc*_label_prob_{}.csv'.format(XV_set))
    files_found = glob.glob(search_path)
    files_found.sort()
    fn0 = files_found[0]
    df = pd.read_csv(fn0,index_col=0)
    print('Generating the prediction accuracy for training files starting with : {} ... '.format(fn0))
    for fn in files_found[1:]:
        # print('Adding file : {}'.format(os.path.basename(fn)))
        dfi = pd.read_csv(fn,index_col=0)
        df = df.append(dfi,ignore_index=True)
    return df,len(files_found)


def summarize_df_split(df,XV_set='valid'):
    cols = list(df)
    path_col = 'path {}'.format(XV_set)
    cols.remove(path_col)
    gk = df.groupby([path_col])

    paths = []
    st1_lbl_pred_all = []

    for name, group in gk: #list(gk)[:20]:
        paths.append(name)
        st1_lbl_pred = list(group[cols].mean()) 
        st1_lbl_pred_all.append(st1_lbl_pred)

    d = {'path':paths, 'st1_mean':st1_lbl_pred_all}
    df_summarized = pd.DataFrame(d)
    return df_summarized

def get_clean_prob(st_list):
    st_array = np.array(st_list)
    clean_prob = np.zeros((st_array.shape[0],2))
    clean_prob[:,0] = st_array[:,0]
    clean_idx = st_array.shape[1]//2
    clean_prob[:,1] = st_array[:,clean_idx]
    return clean_prob

def get_metrics(clean_prob_arr,t):
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

def gen_df_clean(df):
    # this is used to compate to thresholds below, indictator function
    col = 'st1_mean'
    st_list = list(df[col])
    #st_list = [item for item in list(df[col])]
    clean_prob_arr = get_clean_prob(st_list)

    thresholds=[0.99999,0.9999,0.999]+[0.01*(99-i) for i in range(99)]+[1e-3,1e-4,1e-5]

    df_clean = pd.DataFrame(columns=['thresholds','accuracy','tpr','fpr','tp','fn','fp','tn'])
    df_clean['thresholds'] = thresholds
    for t in df_clean['thresholds']:
        acc,tpr,fpr,tp,fn,fp,tn = get_metrics(clean_prob_arr,t)
        df_clean.loc[df_clean['thresholds']==t, ['accuracy','tpr','fpr','tp','fn','fp','tn']] = [acc,tpr,fpr,tp,fn,fp,tn]

    df_clean = append_corners(df_clean)
    return df_clean

def get_mean_var(frames):
    nb_frames = len(frames)
    f_size = frames[0].shape
    arr = np.empty((nb_frames,)+f_size)
    for idx,f in enumerate(frames):
        arr[idx] = f.to_numpy()
    arr_mean = np.mean(arr,axis=0)
    arr_var = np.var(arr,axis=0)
    df_mean = pd.DataFrame(arr_mean,columns=list(frames[0]))
    df_var = pd.DataFrame(arr_var,columns=list(frames[0]))
    return df_mean,df_var,nb_frames

def get_5k_avg(df,XV_set):
    df_summarized = summarize_df_split(df,XV_set)
    df_clean = gen_df_clean(df_summarized)
    return df_clean

def plot_5k_avg(df,mylabel='None'):
    plt.plot(df.fpr,df.tpr,'k-',linewidth=2,label=mylabel)

def compute_AUC(df):
    df['dx'] = df['fpr'] - df['fpr'].shift(1)
    df['fxdx'] = df['tpr'] * df['dx']
    AUC = np.sum(df['fxdx'])
    return AUC

def compute_se(df_s,df_ave):
    dFPR = df_s.fpr - df_ave.fpr
    dTPR = df_s.tpr - df_ave.tpr
    SE = dFPR.transform(lambda x:x**2) + dTPR.transform(lambda x:x**2)
    AMP = (df_ave.fpr - df_ave.tpr)/np.sqrt(2.0)
    NRG = AMP.transform(lambda x:x**2)
    NMSE = SE.sum() / NRG.sum()
    return SE.sum(),NMSE

def split_df(df,mc_runs):
    print('We will split the {} MC inferences into two'.format(mc_runs))
    mc_runs = mc_runs//2
    df_len = df.shape[0]
    df1 = df.iloc[:df_len//2].reset_index(drop=True)
    df2 = df.iloc[df_len//2:].reset_index(drop=True)
    return df1,df2,mc_runs

def plot_mc_roc_splits(df,mc_runs,XV_set='valid',fig_dir='/tmp/'):
    df,df2,mc_runs = split_df(df,mc_runs)
    df_5k = get_5k_avg(df2,XV_set)
    nb_valid = df.shape[0]//mc_runs
    print('Length of validation data subset : {}'.format(nb_valid))
    nb_splits = 50
    fig_label = ['(a)','(b)','(c)']
    # number of inferences to estimate variance of ROC, lets vary
    for idx,nb_inf in enumerate([1,10,100]):
        # randomly shuffle the rows prior to selecting 100 splits
        df = df.sample(frac=1).reset_index(drop=True)
        print('Just shuffled, will select first {} splits after grouping'.format(nb_splits))
        fig_nb = 1001
        plt.figure(fig_nb,figsize=(8,6))
        print('We will split the {} mc_runs by grouping {} inferences'.format(mc_runs,nb_inf))
        frames = []
        sse = []
        nmse = []
        auc = []
        nb_splits_available = mc_runs//nb_inf
        print('Total number of splits available : {}'.format(nb_splits_available))
        if nb_splits_available < nb_splits:
            print('We do not have enough splits, setting nb_splits to {}'.format(nb_splits_available))
            nb_splits = nb_splits_available

        for s in range(nb_splits):
            print('Working on split number : {}'.format(s))
            start = nb_valid*s*nb_inf
            end = nb_valid*(s+1)*nb_inf
            df_split = df.iloc[start:end,:]
            df_split_summarized = summarize_df_split(df_split,XV_set)
            df_clean = gen_df_clean(df_split_summarized)
            frames += [df_clean]
            mylabel = '_nolegend_'
            if not s:
                mylabel = 'ROC from {} MC inference(s)'.format(nb_inf)
            plt.plot(df_clean.fpr,df_clean.tpr,'--',linewidth=0.2,label=mylabel)
            sse_split,nmse_split = compute_se(df_clean,df_5k)
            sse += [sse_split]
            nmse += [nmse_split]
            auc_split = compute_AUC(df_clean)
            auc += [auc_split]
            # print('SSE of split : {}'.format(sse_split))
        # textstr='For {0} ROC curves shown\nAUC : {1:0.3f} ± {2:0.3f}\nSSE : {3:0.3f} ± {4:0.3f}\nNMSE : {5:0.3f} ± {6:0.3f} %'.format(nb_splits,np.mean(auc),np.std(auc),np.mean(sse),np.std(sse),100.0*np.mean(nmse),100.0*np.std(nmse))
        textstr='For {0} ROC curves shown\nAUC : {1:0.3f} ± {2:0.3f}\nSSE : {3:0.3f} ± {4:0.3f}'.format(nb_splits,np.mean(auc),np.std(auc),np.mean(sse),np.std(sse))
        df_mean,df_var,nb_frames = get_mean_var(frames)
        df_sem = df_var.transform(lambda x:x**0.5)/np.sqrt(nb_frames)
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        # place a text box in upper left in axes coords
        plt.text(0.4, 0.6, textstr, fontsize=17,verticalalignment='top', bbox=props)
        plt.text(0.0,1.0,fig_label[idx],va='top',fontsize=20)
        # plt.fill_between(df_mean.fpr,df_mean.tpr + df_sem.tpr, df_mean.tpr - df_sem.tpr, color='r',alpha=0.3)
        # plt.fill_betweenx(df_mean.tpr,df_mean.fpr + df_sem.fpr, df_mean.fpr - df_sem.fpr, color='r',alpha=0.3)
        # plt.plot(df_mean.fpr,df_mean.tpr,'r-',color=(0.8,0.0,0.0),linewidth=1.5,label='avg {} curves'.format(nb_splits))
        plot_5k_avg(df_5k,mylabel='ROC from {} MC inferences'.format(mc_runs))
        plt.plot([0.0,1.0],[0.0,1.0],'c-',linewidth=2.0,label='naive classifier')
        plt.axis([-0.05, 1.05, -0.05, 1.05])
        plt.tick_params(labelsize=15)
        plt.legend(loc='lower right',fontsize=17)
        plt.xlabel('false positive rate',fontsize=17)
        plt.ylabel('true positive rate',fontsize=17)
        plt.tight_layout()
        plt.grid(True)

        fn = os.path.join(fig_dir,'roc.split_nb_inferences_{0:03d}.png'.format(nb_inf))
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)
        plt.close(fig_nb)


start_time = time.time()

# python noise.mc_summarize_prediction.py rap_NN007_single_artifact/clean_percent_050 XV3 test nb_classes_02 nb_samples_factor_01.00
experiment = sys.argv[1]
XV_nb = sys.argv[2]
XV_set = sys.argv[3]
classes = sys.argv[4]
factor = sys.argv[5]

mc_pred_dir = '/data/rpizarro/noise/prediction/{}/{}/{}/{}/ep0500/{}'.format(experiment,XV_nb,classes,factor,XV_set)
print('Working in the following data dir : {}'.format(mc_pred_dir))


df,mc_runs = join_csv_files(mc_pred_dir,XV_set)

print('Number of times we ran MC inferences : {}'.format(mc_runs))

fig_dir = '/data/rpizarro/noise/figs/performance/mc/{}/{}/{}/{}/{}'.format(experiment,XV_nb,XV_set,classes,factor)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir,exist_ok=True)

plot_mc_roc_splits(df,mc_runs,XV_set,fig_dir)

elapsed_time = time.time() - start_time
print('Time it took to compute ROC curves : {0:0.2f} seconds'.format(elapsed_time))

