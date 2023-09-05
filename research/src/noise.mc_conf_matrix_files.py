import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import os,sys
import itertools
from ast import literal_eval


def get_lbl_pred_idx(st):
    lbl = st[:len(st)//2]
    lbl_index = lbl.index(max(lbl))
    pred = st[len(st)//2:]
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


def get_accuracy_cols(df,XV_set,key='st1_index'):
    path_col = ['path']
    accuracy_df = pd.DataFrame(columns=[path_col[0],'lbl_category','NN_category','match_category'])
    accuracy_df[path_col] = df[path_col]
    cols = ['clean','intensity','motion','coverage']

    for index, row in df.iterrows():
        st = literal_eval(row[key])
        lbl_index,pred_index = get_lbl_pred_idx(st)
        append_row = [cols[lbl_index],cols[pred_index],lbl_index==pred_index]
        accuracy_df.loc[accuracy_df['path']==row['path'],['lbl_category','NN_category','match_category']] = append_row

    return accuracy_df,path_col,cols


# Usage
# python noise.conf_matrix_files.py rap_NN007_100ep_CLR test
# choices : ['clean', 'intensity', 'motion-ringing', 'coverage']
experiment = sys.argv[1]
XV_set = sys.argv[2]

sheets_dir = '/home/rpizarro/noise/prediction/{}/'.format(experiment)
fn = os.path.join(sheets_dir,'{}_stats_summarized.csv'.format(XV_set))
print('Loading file : {}'.format(fn))
df = pd.read_csv(fn,index_col=0)


accuracy_df,path_col,cols = get_accuracy_cols(df,XV_set,key='st1_index')

conf = compute_confusion(df,col='st1_index')
print('\nCorresponding confusion matrix with row,col:\n {}'.format(cols))
print(conf)

conf = compute_conf(accuracy_df,cols)
print('\nCorresponding confusion matrix with row,col:\n {}'.format(cols))
print(conf)

for target in cols:
    for inferred in cols:
        print('Looking for files that are (target,inferred) : ({},{})'.format(target,inferred))
        
        files_list = match_files(accuracy_df,path_col,target,inferred)
        if not len(files_list):
            print('No matching files for this pair')
            continue
        print(files_list)

        parent_dir = os.path.join('/home/rpizarro/noise/sheets/confusion_selection/',experiment)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        save_dir = os.path.join(parent_dir,XV_set)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fn = os.path.join(save_dir,'tgt-{}_inf-{}.csv'.format(target,inferred))
        print('Saving matched files to : {}'.format(fn))
        files_list.to_csv(fn,header='path')


