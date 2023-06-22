import pandas as pd
import os,sys
from ast import literal_eval
import numpy as np


def get_clean_prob(st_list):
    st_array = np.array(st_list)
    clean_prob = np.zeros((st_array.shape[0],2))
    clean_prob[:,0] = st_array[:,0]
    clean_idx = st_array.shape[1]//2
    clean_prob[:,1] = st_array[:,clean_idx]
    return clean_prob


def get_metrics(clean_prob_arr,t):
    # column 0 is ground truth and it's either 0 or 1, so we compare to 0.5
    # tp is correctly identified artifact 
    tp = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] <= t))
    # fn is artifact classified as clean, try to minimize
    fn = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] > t))
    # fp is clean classified as artifact
    fp = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] <= t))
    # tn is correctly identified clean
    tn = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] > t))
    # tp = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] > t))
    # fn = np.sum((clean_prob_arr[:,0]>0.5)&(clean_prob_arr[:,1] <= t))
    # fp = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] > t))
    # tn = np.sum((clean_prob_arr[:,0]<0.5)&(clean_prob_arr[:,1] <= t))

    eps = sys.float_info.epsilon
    acc = 1.0*(tp+tn)/(tp+tn+fp+fn+eps)
    tpr = 1.0*tp/(tp+fn+eps)
    fpr = 1.0 - 1.0*tn/(tn+fp+eps)
    return acc,tpr,fpr,tp,fn,fp,tn

def rescale_unc(df,unc_cols = ['MI','entropy','sample_variance']):
    for c in unc_cols:
        a = df[c].min()
        b = df[c].max()
        df[c] = (df[c]-a)/(b-a)
    return df




def get_clean_prob_arr(df,eta):
    df_eta = df.loc[df['sample_variance']<=eta]
    col = 'st1_mean'
    st_list = [literal_eval(item) for item in list(df_eta[col])]
    clean_prob_arr = get_clean_prob(st_list)
    return clean_prob_arr



clean_percent = sys.argv[1]
XV_set = sys.argv[2]
sub_experiment = sys.argv[3]


experiment = 'rap_NN008_multiple_artifact/{}'.format(clean_percent)
XV_nb = 'XV0'
classes = 'nb_classes_02' 
factor = 'nb_samples_factor_01.00'


pred_dir = '/home/rpizarro/noise/prediction/'
mc_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor)
sub_x_dir = os.path.join(mc_dir,sub_experiment)

fn = os.path.join(sub_x_dir,'{}_stats_summarized.csv'.format(XV_set))
print('Reading file : {}'.format(fn))
unc_cols = ['MI','entropy','sample_variance']
df = pd.read_csv(fn,index_col=0)
df = rescale_unc(df,unc_cols)
print(df[['st1_mean','sample_variance']])
print(df['sample_variance'].min())
print(df['sample_variance'].max())
# sys.exit()


set_sz = df.shape[0]

fn = os.path.join(sub_x_dir,'summary_stats.threshold_by_unc.csv')
df_valid = pd.read_csv(fn,index_col=0)


unc_metric = ['MI','entropy','sample_variance']


df_stats = pd.DataFrame(columns=['uncertainty','eta','retained','thresh_maxJ','accuracy','specificity','tn','fp','sensitivity','fn','tp','DFFMR','UDM'])
cols = list(df_stats)

for c in unc_metric:
    print('Currently working on uncertainty metric : {}'.format(c))
    dc = df_valid.loc[df_valid['unc_metric']==c].reset_index(drop=True)
    print(list(dc))
    for index, row in dc.iterrows():
        t = row['thresh_maxJ']
        eta = row['eta_thresh']
        clean_prob_arr = get_clean_prob_arr(df,eta)
        retained_nb = len(clean_prob_arr)
        print('After uncertainty eta<{} threshold we have {} left'.format(eta,retained_nb))
        acc,tpr,fpr,tp,fn,fp,tn = get_metrics(clean_prob_arr,t)
        DFFMR_nb = (set_sz - retained_nb) + tp + fp
        DFFMR = 100.0 * DFFMR_nb / set_sz
        UDM = 100.0 * fn / set_sz
        r1_values = [c,eta,100.0*retained_nb/set_sz,t,acc,1-fpr,tn,fp,tpr,fn,tp,DFFMR,UDM]
        r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
        row1 = pd.DataFrame(r1_dict,index=[0])
        df_stats = df_stats.append(row1,ignore_index=True)
        print('For Youden {0:0.2f}, test metrics are: (acc,sens,spec,tp,fn,fp,tn):({1:0.3f},{2:0.3f},{3:0.3f},{4},{5},{6},{7})'.format(t,acc,tpr,1-fpr,tp,fn,fp,tn))
        print('Corresponding BBK metrics: (DFFMR,UDM) : ({0:0.1f},{1:0.3f})\n'.format(DFFMR,UDM))



print(df_stats)
fn = os.path.join(sub_x_dir,'summary_stats.threshold_by_unc.test.csv')
print('Saving stas to : {}'.format(fn))
df_stats.to_csv(fn)



