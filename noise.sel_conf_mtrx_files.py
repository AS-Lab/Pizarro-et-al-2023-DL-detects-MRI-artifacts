import numpy as np
np.set_printoptions(suppress=True)
import os,sys
import itertools
from subprocess import Popen, PIPE, call
import pandas as pd
pd.options.display.width = 0

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


def compute_conf(accuracy_df,cols):

    confusion_matrix = np.zeros((len(cols),len(cols)))

    for ci,c in enumerate(cols):
        for di,d in enumerate(cols):
            confusion_matrix[ci,di] = ( (accuracy_df['lbl_category']==c)
                    & (accuracy_df['NN_category']==d) ).sum()


    return np.transpose(confusion_matrix)


def check_input(cols,col1,col2):
    if col1 not in cols:
        print('Target not one of the choices : {}'.format(col1))
        print(cols)
        sys.exit()
    if col2 not in cols:
        print('Inferred not one of the choices : {}'.format(col2))
        print(cols)
        sys.exit()
    if col1 == col2:
        print('Please select different values of the choices:')
        print(cols)
        sys.exit()


def get_file_list(df_XV_set,experiment,XV_set,c1,c2,n):
    parent_dir = '/home/rpizarro/noise/sheets/confusion_selection/'
    XV_set_dir = os.path.join(parent_dir,experiment,XV_set)
    c12_fn = os.path.join(XV_set_dir,'tgt-{}_inf-{}.csv'.format(c1,c2))
    c21_fn = os.path.join(XV_set_dir,'tgt-{}_inf-{}.csv'.format(c2,c1))

    if not os.path.isfile(c12_fn):
        print('We could not find file : {}'.format(c12_fn))
        sys.exit()
    if not os.path.isfile(c21_fn):
        print('We could not find file : {}'.format(c21_fn))
        sys.exit()


    df12 = pd.read_csv(c12_fn,index_col=0)
    df21 = pd.read_csv(c21_fn,index_col=0)

    if len(df12) < n:
        print('We did not find enough (tgt,inf) : ({},{})'.format(c1,c2))
        print('Instead of {} we will look for {} files'.format(n,len(df12)))
        n = len(df12)
    if len(df21) < n:
        print('We did not find enough (tgt,inf) : ({},{})'.format(c2,c1))
        print('Instead of {} we will look for {} files'.format(n,len(df21)))
        n = len(df21)

    df12_sample = df12.sample(n=n).reset_index(drop=True)
    df12_sample['target'] = c1
    df21_sample = df21.sample(n=n).reset_index(drop=True)
    df21_sample['target'] = c2
    df = pd.concat([df12_sample,df21_sample],ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    s_arr = list(pd.util.testing.rands_array(10, len(df)))
    data_link_dir = '/data/datasets/shared/rpizarro/noise/data_copy'
    fn_arb = [os.path.join(data_link_dir,s+'.mnc.gz') for s in s_arr]
    df['path arbitraty'] = fn_arb

    lbl_NN_cols = list(df_XV_set)[2:-1]
    print(lbl_NN_cols)
    # for c in lbl_NN_cols:
    #     df[c]=0
    # print(df.head())
    df = pd.concat([df,pd.DataFrame(columns=lbl_NN_cols)],sort=False)
    for index,row in df.iterrows():
        for c in lbl_NN_cols:
            df.loc[df['path test']==row['path test'],c] = float(df_XV_set.loc[df_XV_set['path test']==row['path test'],c])
        cmd_link = 'ln -s {} {}'.format(row['path test'],row['path arbitraty'])
        prd= Popen(cmd_link.split(' '))
        prd.communicate()

    save_dir = os.path.join(XV_set_dir,'select_few')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fn = os.path.join(save_dir,'{}_{}_target_revealed.csv'.format(c1,c2))
    print('Saving paths with targets to : {}'.format(fn))
    df.to_csv(fn)

    df = df[['path arbitraty']]
    # df = df.drop(columns=['target','path test'])
    fn = os.path.join(save_dir,'{}_{}.csv'.format(c1,c2))
    print('Saving paths to : {}'.format(fn))
    df.to_csv(fn)

    return df


# Usage
# python noise.sel_conf_mtrx_files.py rap_NN007_100ep_CLR test clean intensity number
# choices : ['clean', 'intensity', 'motion-ringing', 'coverage']
experiment = sys.argv[1]
XV_set = sys.argv[2]
col1 = sys.argv[3]
col2 = sys.argv[4]
nb_files = int(sys.argv[5])

sheets_dir = '/home/rpizarro/noise/prediction/{}/'.format(experiment)
fn = os.path.join(sheets_dir,'label_prob_{}.csv'.format(XV_set))
df = pd.read_csv(fn,index_col=0)
accuracy_df,path_col,cols = get_accuracy_cols(df,XV_set)
check_input(cols,col1,col2)

conf = compute_conf(accuracy_df,cols)
print('\nCorresponding confusion matrix with row,col:\n {}'.format(cols))
print(conf)

file_list = get_file_list(df,experiment,XV_set,col1,col2,nb_files)

# print(file_list)







