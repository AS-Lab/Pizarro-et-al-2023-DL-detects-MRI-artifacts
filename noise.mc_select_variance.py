import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import os,sys
import itertools
from ast import literal_eval
pd.set_option('display.max_rows', 200)

def get_lbl_pred(st):
    lbl = st[:len(st)//2]
    pred = st[len(st)//2:]
    return lbl,pred

def get_lbl_pred_idx(st):
    lbl,pred=get_lbl_pred(st)
    lbl_index = lbl.index(max(lbl))
    pred_index = pred.index(max(pred))
    return lbl_index,pred_index


def compute_var(df,col='st1'):
    conf_index = compute_confusion(df,col = col+'_index')
    col_var = col + '_var'
    col_index = col + '_index'
    st_list = [literal_eval(item) for item in list(df[col_index])]
    nb_out = len(st_list[0])//2
    conf_var = np.zeros((nb_out,nb_out))
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st)
        st_var = literal_eval(list(df.iloc[[index]][col_var])[0])
        conf_var[pred_index,lbl_index] += st_var[pred_index+nb_out]
    conf_var = np.divide(conf_var,conf_index)
    return conf_var



def compute_mean(df,col='st1'):
    conf_index = compute_confusion(df,col = col+'_index')
    col_mean = col + '_mean'
    col_index = col + '_index'
    st_list = [literal_eval(item) for item in list(df[col_index])]
    nb_out = len(st_list[0])//2
    conf_mean = np.zeros((nb_out,nb_out))
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st)
        st_mean = literal_eval(list(df.iloc[[index]][col_mean])[0])
        conf_mean[pred_index,lbl_index] += st_mean[pred_index+nb_out]
    conf_mean = np.divide(conf_mean,conf_index)
    return conf_mean


def compute_confusion(df,col='st1_index'):
    st_list = [literal_eval(item) for item in list(df[col])]
    nb_out = len(st_list[0])//2
    conf = np.zeros((nb_out,nb_out))
    for index,st in enumerate(st_list):
        lbl_index,pred_index = get_lbl_pred_idx(st)
        conf[pred_index,lbl_index] += 1
    return conf




def match_files(df,path_col,tgt,inf):
    files = df[path_col].where((df['lbl_category']==tgt) & (df['NN_category']==inf)).dropna().reset_index(drop=True)
    return files

def compute_conf(accuracy_df,cols):

    confusion_matrix = np.zeros((len(cols),len(cols)))

    for ci,c in enumerate(cols):
        for di,d in enumerate(cols):
            confusion_matrix[ci,di] = ( (accuracy_df['lbl_category']==c)
                    & (accuracy_df['NN_category']==d) ).sum()


    return np.transpose(confusion_matrix)


def get_variance(df,XV_set,key='st1'):
    path_col = ['path']
    variance_df = pd.DataFrame(columns=[path_col[0],'lbl_category','NN_category','NN_variance','NN_mean','match_category'])
    variance_df[path_col] = df[path_col]
    cols = ['clean','intensity','motion','coverage']

    for index, row in df.iterrows():
        st_index = literal_eval(row[key+'_index'])
        lbl_index,pred_index = get_lbl_pred_idx(st_index)
        st_var = literal_eval(row[key+'_var'])
        lbl_var,pred_var = get_lbl_pred(st_var)
        st_mean = literal_eval(row[key+'_mean'])
        lbl_mean,pred_mean = get_lbl_pred(st_mean)
        append_row = [cols[lbl_index],cols[pred_index],pred_var[pred_index],pred_mean[pred_index],lbl_index==pred_index]
        variance_df.loc[variance_df['path']==row['path'],['lbl_category','NN_category','NN_variance','NN_mean','match_category']] = append_row

    return variance_df,path_col,cols


# Usage
# python noise.conf_matrix_files.py rap_NN007_100ep_CLR test
# choices : ['clean', 'intensity', 'motion-ringing', 'coverage']
experiment = sys.argv[1]
XV_set = sys.argv[2]

sheets_dir = '/home/rpizarro/noise/prediction/{}/'.format(experiment)
fn = os.path.join(sheets_dir,'{}_stats_summarized.csv'.format(XV_set))
print('Loading file : {}'.format(fn))
df = pd.read_csv(fn,index_col=0)


variance_df,path_col,cols = get_variance(df,XV_set,key='st1')

# variance_df.sort_values(by=['NN_variance'],ascending=False,inplace=True)

# print(variance_df[:200])


parent_dir = os.path.join('/home/rpizarro/noise/sheets/confusion_selection/',experiment)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)
save_dir = os.path.join(parent_dir,XV_set)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fn = os.path.join(save_dir,'{}_st1_prediction_variance_probability.csv'.format(XV_set))
print('Saving variance results to : {}'.format(fn))
variance_df.to_csv(fn)
