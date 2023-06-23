import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import os,sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools



def plot_conf(conf,base,mods,t_str):
    fn=os.path.join(base,t_str+'.png')
    # fn='./figs/confusion/modalities{}_f002.pdf'.format(len(mods))
    cmap=plt.cm.Blues
    fnt_size = 13
    if conf.shape[0]<3:
        fnt_size = 20
    fig=plt.figure(figsize=(5,5))
    # imgplot=plt.imshow(conf)
    imgplot=plt.imshow(conf, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(mods))
    cbar = plt.colorbar(imgplot,fraction=0.046, pad=0.04) # ticks=[0,10,20,30,40,50])
    # cbar.ax.set_yticklabels(['0','','','','','50'])
    cbar.ax.tick_params(labelsize=fnt_size)
    plt.xticks(tick_marks, mods, rotation=45, fontsize=fnt_size)
    plt.yticks(tick_marks, mods, fontsize=fnt_size)
    # plt.figure()
    # plt.colorbar()
    #plt.imshow(conf_5mod)
    # plt.imsave(fn,np.asarray(conf_5mod))
    # plt.title(t_str.replace('_',' '))
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
    print(fn)
    # fig.savefig(fn, format='eps', dpi=1000)
    #v fig.savefig(fn, format='pdf', bbox_inches='tight',dpi=1000)
    fig.savefig(fn,bbox_inches='tight')
    plt.close


def compute_conf(df):
    lbl_NN_cols = list(df)[2:-1]
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
        for di,d in enumerate(cols):
            confusion_matrix[ci,di] = ( (accuracy_df['lbl_category']==c)
                    & (accuracy_df['NN_category']==d) ).sum()


    print('\nCorresponding confusion matrix with row,col:\n {}'.format(cols))
    print(confusion_matrix)
    return confusion_matrix


# python noise.train_NN.py rap_NN007_100ep_CLR test
experiment = sys.argv[1]
XV_set = sys.argv[2]

sheets_dir = '/home/rpizarro/noise/prediction/{}/prob_level_appended'.format(experiment)
fn = os.path.join(sheets_dir,'label_prob_{}.csv'.format(XV_set))

df = pd.read_csv(fn,index_col=0)

conf = compute_conf(df)

figs_dir = '/home/rpizarro/noise/figs/confusion/'
outcomes_str = ['clean','intensity','motion','coverage']
title = '{}_{}'.format(experiment,XV_set)

plot_conf(np.transpose(conf),figs_dir,outcomes_str,title)

conf_2x2 = np.zeros((2,2))
conf_2x2[0,0] = conf[0,0]
conf_2x2[0,1] = np.sum(conf[0,1:])
conf_2x2[1,0] = np.sum(conf[1:,0])
conf_2x2[1,1] = np.sum(conf[1:,1:])
print(conf_2x2)

outcomes_str = ['clean','artifact']
title += '_2x2'
plot_conf(np.transpose(conf_2x2),figs_dir,outcomes_str,title)





