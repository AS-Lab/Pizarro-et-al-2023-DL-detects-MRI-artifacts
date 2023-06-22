import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import os,sys
import itertools


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


def get_accuracy_cols(df,XV_set):
    cols_all = list(df)
    path_col = [c for c in cols_all if XV_set in c]
    accuracy_df = pd.DataFrame(columns=[path_col[0],'lbl_category','NN_category','match_category'])
    accuracy_df[path_col] = df[path_col]
    lbl_NN_cols = list(df)[2:-1]
    lbl_cols = lbl_NN_cols[:len(lbl_NN_cols)//2]
    cols = [l.replace('lbl_','') for l in lbl_cols]
    lbl_rename = {a:b for a,b in zip(lbl_cols,cols)}
    lbl_df = df[lbl_cols].rename(columns=lbl_rename)
    accuracy_df['lbl_category'] = lbl_df.idxmax(axis=1)

    NN_cols = lbl_NN_cols[len(lbl_NN_cols)//2:]
    NN_rename = {a:b for a,b in zip(NN_cols,cols)}
    NN_df = df[NN_cols].rename(columns=NN_rename)
    accuracy_df['NN_category'] = NN_df.idxmax(axis=1)

    accuracy_df['match_category'] = lbl_df.idxmax(axis=1).eq(NN_df.idxmax(axis=1))

    return accuracy_df,path_col,cols


# Usage
# python noise.conf_matrix_files.py rap_NN007_100ep_CLR test
# choices : ['clean', 'intensity', 'motion-ringing', 'coverage']
experiment = sys.argv[1]
XV_set = sys.argv[2]

sheets_dir = '/home/rpizarro/noise/prediction/{}/'.format(experiment)
fn = os.path.join(sheets_dir,'label_prob_{}.csv'.format(XV_set))
print('Loading file : {}'.format(fn))
df = pd.read_csv(fn,index_col=0)
accuracy_df,path_col,cols = get_accuracy_cols(df,XV_set)

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


