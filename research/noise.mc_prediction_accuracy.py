import pandas as pd
import glob
import numpy as np
import os,sys
np.set_printoptions(suppress=True)

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

    acc = 1.0*(tp+tn)/(tp+tn+fp+fn)
    tpr = 1.0*tp/(tp+fn)
    tnr = 1.0*tn/(tn+fp)
    return acc,tpr,tnr

def print_prompt():
    print('\nArtifact detection means that positive there IS specified category:\n')
    print('tp - true positive : there is category and NN detected category')
    print('fn - false negative: there is category but NN detected NO category')
    print('fp - false positive: there is NO category but NN detected category')
    print('tn - true negative : there is NO category and NN detected NO category\n')


def get_prediction_accuracy(fn):
    df = pd.read_csv(fn,index_col=0)
    print('Generating the prediction accuracy for : {}'.format(fn))

    lbl_NN_cols = list(df)[:-1]
    lbl_cols = lbl_NN_cols[:len(lbl_NN_cols)//2]
    cols = [l.replace('lbl_','') for l in lbl_cols]
    lbl_rename = {a:b for a,b in zip(lbl_cols,cols)}
    lbl_df = df[lbl_cols].rename(columns=lbl_rename)
    accuracy_df = pd.DataFrame(columns=['lbl_category'])
    accuracy_df['lbl_category'] = lbl_df.idxmax(axis=1)

    NN_cols = lbl_NN_cols[len(lbl_NN_cols)//2:]
    NN_rename = {a:b for a,b in zip(NN_cols,cols)}
    NN_df = df[NN_cols].rename(columns=NN_rename)
    accuracy_df['NN_category'] = NN_df.idxmax(axis=1)

    accuracy_df['match_category'] = lbl_df.idxmax(axis=1).eq(NN_df.idxmax(axis=1))

    confusion_matrix = np.zeros((len(cols),len(cols)))

    for ci,c in enumerate(cols):
        if 'clean' in c:
            print('\nComputing sensitivity and specificity for >>>{}<<<'.format(c))
            acc,tpr,tnr = calculate_metrics(accuracy_df,c)
            print('Overall accuracy ; (tp+tn)/(tp+tn+fp+fn) = {0:0.3f}'.format(acc))
            print('Sensitivity - true positive rate : tp/(tp+fn) = {0:0.3f}'.format(tpr))
            print('Specificity - true negative rate : tn/(tn+fp) = {0:0.3f}'.format(tnr))
        for di,d in enumerate(cols):
            confusion_matrix[ci,di] = ( (accuracy_df['lbl_category']==c)
                    & (accuracy_df['NN_category']==d) ).sum()

    print('\nCorresponding confusion matrix with row,col:\n {}'.format(cols))

    print(confusion_matrix)


# python noise.train_NN.py rap_NN007_100ep_CLR test
experiment = sys.argv[1]
XV_set = sys.argv[2]

print_prompt()

csv_dir = '/home/rpizarro/noise/prediction/{}/repro/{}'.format(experiment,XV_set)
search_path = os.path.join(csv_dir,'mc*label_prob_{}.csv'.format(XV_set))
fn_list = glob.glob(search_path)
fn_list.sort()

for fn in fn_list:
    # print(fn)
    get_prediction_accuracy(fn)


