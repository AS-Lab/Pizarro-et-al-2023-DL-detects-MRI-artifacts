import time
from pathlib import Path
import pandas as pd
import os,sys
import glob
from ast import literal_eval
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools


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


def append_corners(df):
    cols = list(df)
    nb_pos = df.iloc[0].tp + df.iloc[0].fn
    nb_neg = df.iloc[0].fp + df.iloc[0].tn

    r1_values = [1.0,0.0,1.0,1.0,nb_pos,0,nb_neg,0]
    r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
    row1 = pd.DataFrame(r1_dict,index=[0])
    df = row1.append(df,ignore_index=True)

    rN_values = [0.0,0.0,0.0,0.0,0,nb_pos,0,nb_neg]
    rN_dict = {k:v for (k,v) in zip(cols,rN_values)}
    rowN = pd.DataFrame(rN_dict,index=[0])
    df = df.append(rowN,ignore_index=True)

    return df

def plot_conf(conf,outcomes=['clean','artifact'],index_mode=True):
    cmap=plt.cm.Blues
    fnt_size = 25
    if conf.shape[0]<3:
        fnt_size = 35
    # fig=plt.figure(fig_nb,figsize=(5,5))
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
        if (not index_mode) and (conf[i,j]<1.0):
            plt.text(j, i, '{0:0.3f}'.format(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")
        elif conf[i,j]>0:
            plt.text(j, i, int(conf[i, j]),
                    horizontalalignment="center",fontsize=fnt_size,
                    color="white" if conf[i, j] > thresh else "black")


def plot_roc_conf(save_dir,df,conf,t):
    fig_nb = int(t*10000)
    plt.figure(fig_nb,figsize=(20,10))

    plt.subplot(121)
    plt.plot(df.fpr,df.tpr)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    # error = np.sqrt( df['tpr_std']**2 + df['fpr_std']**2 )
    # plt.fill_between(df.fpr_mean, df.tpr_mean-df.tpr_std, df.tpr_mean+df.tpr_std, interpolate=True,alpha=0.1)
    # plt.fill_betweenx(df.tpr_mean, df.fpr_mean-df.fpr_std, df.fpr_mean+df.fpr_std, interpolate=True,alpha=0.1)

    fpr_threshold = float(df[df['thresholds']==t]['fpr'])
    tpr_threshold = float(df[df['thresholds']==t]['tpr'])
    plt.plot(fpr_threshold,tpr_threshold,'r*',markersize=20)
    plt.text(0.6,0.4,'t={0:0.3f}'.format(t),fontsize=17)
    plt.tick_params(labelsize=15)
    plt.xlabel('false positive rate',fontsize=17)
    plt.ylabel('true positive rate',fontsize=17)
    plt.tight_layout()
    plt.grid(True)

    plt.subplot(122)
    plot_conf(conf,outcomes=['clean','artifact'],index_mode=True)

    fn = os.path.join(save_dir,'clean_t{0:0.5f}.roc_conf.png'.format(t))
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def compare_threshold_by_unc(outfile,save_dir,df,unc_cols = ['MI','entropy','sample_variance']):
    # unc_thresholds=[0.99,0.95,0.9,0.7,0.5,0.2,0.1,0.05,0.01,0.001]
    unc_thresholds=[0.5,0.1,0.01,0.001]
    df = rescale_unc(df,unc_cols)
    colors = ['k','r','g','b','m','c']+['k']*6
    XVset_length = df.shape[0]
    df_stats = pd.DataFrame(columns=['XVset','eta_thresh','unc_metric','retained','AUC','thresh_maxJ','accuracy','sensitivity','specificity','tp','fn','fp','tn'])
    for c in unc_cols:
        fig_nb = 1234
        # markersize
        ms = 10
        xJ = []
        yJ = []
        plt.figure(fig_nb,figsize=(8,6))
        
        print('Baseline no eta threshold')
        df_base = gen_df_clean(df)
        print_naive_classifier(df_base)
        summary_stats = comp_stats(df_base,set_sz=XVset_length,J='J')
        xJ += [1-summary_stats[4]]
        yJ += [summary_stats[3]]
        df_stats = append_stats(df_stats,df.XV_nb,summary_stats,nb_epoch=df.nb_epoch,eta=1.0,unc=c,retained=100)
        plt.plot(df_base.fpr,df_base.tpr,'{}-'.format(colors[0]),linewidth=2.0,label='100%, none')
        plt.plot(1-summary_stats[4],summary_stats[3],'{}*'.format(colors[0]),markersize=ms)
        plt.axis([-0.05, 1.05, -0.05, 1.05])
        percent_retained = [100]

        for idx,eta in enumerate(unc_thresholds):
            print('(eta,unc_metric) : ({0},{1})'.format(eta,c))
            df_eta = df[df[c]<eta]
            print(df_eta.shape)
            retained = 100.0*df_eta.shape[0]/XVset_length
            percent_retained = percent_retained + [retained]
            print('Percent retained : {0:0.2f}%'.format(retained))
            df_clean = gen_df_clean(df_eta)
            # AUC,threshold,accuracy,sensitivity(tpr),specificity(1-fpr),tp,fn,fp,tn
            summary_stats = comp_stats(df_clean,set_sz=XVset_length,J='J')
            xJ += [1-summary_stats[4]]
            yJ += [summary_stats[3]]
            df_stats = append_stats(df_stats,df.XV_nb,summary_stats,nb_epoch=df.nb_epoch,eta=eta,unc=c,retained=retained)
            plt.plot(df_clean.fpr,df_clean.tpr,'{}-'.format(colors[idx+1]),linewidth=1.0,label=r'{0:0.1f}%, $\sigma^2$<{1}'.format(retained,eta))
            plt.plot(1-summary_stats[4],summary_stats[3],'{}*'.format(colors[idx+1]),markersize=ms)

        print(percent_retained)
        plt.plot([0.0,1.0],[0.0,1.0],'c-',linewidth=2.0,label='naive classifier')
        BBK_sens = 0.91
        BBK_spec = 0.84
        # plt.plot(1-BBK_spec,BBK_sens,'cX',markersize=10,label='Biobank')
        plt.tick_params(labelsize=15)
        legend = plt.legend(title=r'% retained, $\eta$ threshold',title_fontsize=17,loc='lower right',fontsize=17)
        plt.xlabel('false positive rate',fontsize=17)
        plt.ylabel('true positive rate',fontsize=17)
        # plt.title(os.path.basename(save_dir),fontsize=20)
        plt.tight_layout()
        plt.grid(True)

        fn = os.path.join(save_dir,'clean_unc_{0}.roc_unc.global.png'.format(c))
        outfile.write('Saving to : {}\n'.format(fn))
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)

        ax = plt.gca()
        # stage 2
        # plt.axis([-0.005, 0.055, 0.945, 1.005])
        # stage 3 ramping
        plt.axis([-0.005, 0.06, 0.895, 0.935])
        ax.xaxis.tick_top()
        # plt.xticks([0.05,0.1],(0.05,0.1),fontsize=40)
        plt.xticks([0.0,0.05],(0.0,0.05),fontsize=40)
        plt.yticks([0.9,0.93],('0.9','0.93'),fontsize=40)
        for (x,y,clr) in zip(xJ,yJ,colors):
            print(x,y,clr)
            plt.plot(x,y,'{}*'.format(clr),markersize=4*ms)
        # plt.plot(1-BBK_spec,BBK_sens,'cX',markersize=40,label='Biobank')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.tight_layout()
        legend.remove()
        fn = os.path.join(save_dir,'clean_unc_{0}.roc_unc.zoom.png'.format(c))
        outfile.write('Saving to : {}\n'.format(fn))
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)

        plt.close(fig_nb)

    # save stats to csv
    fn = os.path.join(df.pred_dir,'{}'.format('summary_stats.threshold_by_unc.csv'))
    outfile.write('Saving to : {}\n'.format(fn))
    print('Saving to : {}'.format(fn))
    df_stats.to_csv(fn)



def append_stats(df,c1,summary_stats,nb_epoch='ep0100',eta=-1.0,unc='baseline',retained=100):
    # c1 is column 1 used for comparison
    AUC,thres_maxJ,accuracy,sensitivity,specificity,tp,fn,fp,tn = summary_stats
    cols = list(df)
    if eta<0:
        r1_values = [c1] + [nb_epoch] + list(summary_stats)
    else:
        r1_values = [c1,eta,unc,retained,AUC,thres_maxJ,accuracy,sensitivity,specificity,tp,fn,fp,tn]
    r1_dict = {k:v for (k,v) in zip(cols,r1_values)}
    row1 = pd.DataFrame(r1_dict,index=[0])
    df_stats = df.append(row1,ignore_index=True)
    return df_stats


# def compare_unc_by_threshold(outfile,save_dir,df,unc_thresholds=[0.5,0.1,0.01]):
#     # this should generate three ROC plots comparing UNC to baseline
#     unc_cols = ['MI','entropy','sample_variance']
#     df = rescale_unc(df,unc_cols)
#     colors = ['r','g','b']
#     XVset_length = df.shape[0]
#     df_stats = pd.DataFrame(columns=['XVset','eta_thresh','unc_metric','retained','AUC','thresh_maxJ','accuracy','sensitivity','specificity'])
# 
#     for eta in unc_thresholds:
#         fig_nb = 1234
#         plt.figure(fig_nb,figsize=(8,6))
#         
#         print('Baseline no eta threshold')
#         df_base = gen_df_clean(df)
#         print_naive_classifier(df_base)
# 
#         plt.plot(df_base.fpr,df_base.tpr,'k-',label='Baseline (100%)')
#         plt.axis([-0.05, 1.05, -0.05, 1.05])
#         percent_retained = [100]
# 
#         summary_stats = comp_stats(df_base)
#         df_stats = append_stats(df_stats,df.XV_nb,summary_stats,eta,'baseline','100')
# 
# 
#         for cidx,c in enumerate(unc_cols):
#             print('(eta,unc_metric) : ({0},{1})'.format(eta,c))
#             df_eta = df[df[c]<eta]
#             print(df_eta.shape)
#             retained = 100.0*df_eta.shape[0]/XVset_length
#             percent_retained = percent_retained + [retained]
#             print('Percent retained : {0:0.2f}%'.format(retained))
#             df_clean = gen_df_clean(df_eta)
# 
#             summary_stats = comp_stats(df_clean)
#             df_stats = append_stats(df_stats,df.XV_nb,summary_stats,eta,c,retained)
# 
#             plt.plot(df_clean.fpr,df_clean.tpr,'{}-'.format(colors[cidx]),label='{0} ({1:0.1f}%)'.format(c,retained))
# 
#         print(percent_retained)
#         plt.tick_params(labelsize=15)
#         plt.legend(loc='lower right',fontsize=17)
#         plt.xlabel('false positive rate',fontsize=17)
#         plt.ylabel('true positive rate',fontsize=17)
#         plt.tight_layout()
#         plt.grid(True)
# 
#         fn = os.path.join(save_dir,'clean_eta{0:0.2f}.roc_unc.png'.format(eta))
#         outfile.write('Saving to : {}\n'.format(fn))
#         print('Saving to : {}'.format(fn))
#         plt.savefig(fn)
#         plt.close(fig_nb)
# 
#     # save stats to csv
#     fn = os.path.join(df.pred_dir,'{}'.format('summary_stats.unc_by_threshold.csv'))
#     outfile.write('Saving to : {}\n'.format(fn))
#     print('Saving to : {}'.format(fn))
#     df_stats.to_csv(fn)

def get_conf(row):
    conf = np.zeros((2,2))
    conf[0,0] = row['tp']
    conf[0,1] = row['fp']
    conf[1,0] = row['fn']
    conf[1,1] = row['tn']
    return conf

def plot_roc_conf_threshold(save_dir,df):
    for t in df['thresholds']:
        row = df[df['thresholds']==t]
        conf = get_conf(row)
        plot_roc_conf(save_dir,df,conf,t)

def compute_AUC(df):
    df['j'] = df['tpr'] - df['fpr']
    df['j1'] = 0.1*df['tpr'] - df['fpr']

    df['dx'] = df['fpr'] - df['fpr'].shift(-1)
    df['fxdx'] = df['tpr'] * df['dx']
    AUC = np.sum(df['fxdx'])

    df['dx'] = -1*df['fpr'] + df['fpr'].shift(1)
    df['fxdx'] = df['tpr'] * df['dx']
    AUC1 = np.sum(df['fxdx'])

    df['dy'] = df['tpr'] - df['tpr'].shift(-1)
    df['fydy'] = (1-df['fpr']) * df['dy']
    AUCy = np.sum(df['fydy'])

    df['dy'] = -1*df['tpr'] + df['tpr'].shift(1)
    df['fydy'] = (1-df['fpr']) * df['dy']
    AUCy1 = np.sum(df['fydy'])

    if np.abs(AUCy1 - AUC) > 0.005:
        print('AUC in the x is {} and in the x1 is {}'.format(AUC,AUC1))
        print('AUC in the y is {} and in the y1 is {}'.format(AUCy,AUCy1))
        # AUC = (AUC+AUC2)/2.0
        sys.exit()

    return df,AUC


def print_naive_classifier(df):
    row0 = df[df['thresholds']==1e-5]
    conf = get_conf(row0)
    print('A naive classifier would perform this way')
    print(np.sum(conf,axis=0))


def rescale_unc(df,unc_cols = ['MI','entropy','sample_variance']):
    for c in unc_cols:
        a = df[c].min()
        b = df[c].max()
        df[c] = (df[c]-a)/(b-a)
    return df


def gen_df_clean(df):
    # this is used to compate to thresholds below, indictator function
    col = 'st1_mean'
    st_list = [literal_eval(item) for item in list(df[col])]
    clean_prob_arr = get_clean_prob(st_list)

    thresholds=[0.99999,0.9999,0.999]+[0.01*(99-i) for i in range(99)]+[1e-3,1e-4,1e-5]

    df_clean = pd.DataFrame(columns=['thresholds','accuracy','tpr','fpr','tp','fn','fp','tn'])
    df_clean['thresholds'] = thresholds
    for t in df_clean['thresholds']:
        acc,tpr,fpr,tp,fn,fp,tn = get_metrics(clean_prob_arr,t)
        df_clean.loc[df_clean['thresholds']==t, ['accuracy','tpr','fpr','tp','fn','fp','tn']] = [acc,tpr,fpr,tp,fn,fp,tn]
    
    df_clean = append_corners(df_clean)
    return df_clean

def append_BK(df,set_sz=6958):
    df['retained_nb'] = df['tp']+df['fn']+df['fp']+df['tn']
    set_sz = df['retained_nb'][0]
    df['sent_nb'] = set_sz - df['retained_nb']
    df['class_pos'] = df['tp'] + df['fp']
    df['DFFMR_nb'] = df['sent_nb'] + df['class_pos']
    df['DFFMR_%'] = 100.0 * df['DFFMR_nb'] / set_sz
    df['UDM_%'] = 100.0 * df['fn'] / set_sz
    df['P.prediction_%'] = 100.0 * df['tp'] / (df['tp'] + df['fp'] + 1**(-10))
    df['N.prediction_%'] = 100.0 * df['tn'] / (df['tn'] + df['fn'] + 1**(-10))
    # df['BBK_obj'] = ((df['DFFMR_%']/100.0)**2 + (df['UDM_%']/1.0)**2)**0.5
    df['BBK_obj'] = ((df['DFFMR_%']/17.29)**2 + (df['UDM_%']/0.15)**2)**0.5
    return df


def comp_stats(df_clean,set_sz=6958,J='J'):
    
    df_clean,AUC = compute_AUC(df_clean)
    df_clean = append_BK(df_clean,set_sz=set_sz)
    
    print('The threshold to minimize BBK-error is : ')
    BBK_min = np.argmin(np.array(df_clean['BBK_obj']))
    BBK_min_row = df_clean.iloc[[BBK_min]]
    print(BBK_min_row)
    # tp,fn,fp,tn = (BBK_min_row['tp'],BBK_min_row['fn'],BBK_min_row['fp'],BBK_min_row['tn'])
    tp,fn,fp,tn = (BBK_min_row['tp'].values[0],BBK_min_row['fn'].values[0],BBK_min_row['fp'].values[0],BBK_min_row['tn'].values[0])
    threshold,accuracy,sensitivity,specificity = (BBK_min_row['thresholds'].values[0],BBK_min_row['accuracy'].values[0],BBK_min_row['tpr'].values[0],1-BBK_min_row['fpr'].values[0])
    print("\nAUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn : ({0:0.3f},{1:0.2f},{2:0.3f},{3:0.3f},{4:0.3f},{5},{6},{7},{8})\n".format(AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn))
    if J=='BBK':
        return AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn
    
    print('The threshold to maximize J1-statistic is : ')
    j1_max = np.argmax(np.array(df_clean['j1']))
    j1_max_row = df_clean.iloc[[j1_max]]
    print(j1_max_row)
    tp,fn,fp,tn = (j1_max_row['tp'].values[0],j1_max_row['fn'].values[0],j1_max_row['fp'].values[0],j1_max_row['tn'].values[0])
    threshold,accuracy,sensitivity,specificity = (j1_max_row['thresholds'].values[0],j1_max_row['accuracy'].values[0],j1_max_row['tpr'].values[0],1-j1_max_row['fpr'].values[0])
    print("\nAUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn : ({0:0.3f},{1:0.2f},{2:0.3f},{3:0.3f},{4:0.3f},{5},{6},{7},{8})\n".format(AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn))
    if J=='J1':
        return AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn
    
    print('The threshold to maximize J-statistic is : ')
    j_max = np.argmax(np.array(df_clean['j']))
    j_max_row = df_clean.iloc[[j_max]]
    print(j_max_row)
    tp,fn,fp,tn = (j_max_row['tp'].values[0],j_max_row['fn'].values[0],j_max_row['fp'].values[0],j_max_row['tn'].values[0])
    threshold,accuracy,sensitivity,specificity = (j_max_row['thresholds'].values[0],j_max_row['accuracy'].values[0],j_max_row['tpr'].values[0],1-j_max_row['fpr'].values[0])
    print("\nAUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn : ({0:0.3f},{1:0.2f},{2:0.3f},{3:0.3f},{4:0.3f},{5},{6},{7},{8})\n".format(AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn))
    return AUC,threshold,accuracy,sensitivity,specificity,tp,fn,fp,tn

# def mc_plot_roc_factor(factors,experiment,XV_nb,XV_set,classes,uncertainty=True):
# 
#     fig_nb = 1001
#     plt.figure(fig_nb,figsize=(8,6))
#     colors = ['r','g','k','b','c','m']
#     pred_dir_root = '/home/rpizarro/noise/prediction/{}/{}/{}'.format(experiment,XV_nb,classes)
#     
#     df_stats = pd.DataFrame(columns=['Factor','AUC','thresh_maxJ','accuracy','sensitivity','specificity'])
# 
#     for idx,factor in enumerate(factors):
#         
#         pred_dir = os.path.join(pred_dir_root,factor)
#         fn = os.path.join(pred_dir,'{}.mc_plot_roc.out'.format(XV_set))
#         outfile = open(fn, 'w')
#         
#         fn = os.path.join(pred_dir,'{}_stats_summarized.csv'.format(XV_set))
#         df = pd.read_csv(fn,index_col=0)
# 
#         df.pred_dir = pred_dir
#         df.XV_nb = XV_nb
#         df.experiment = experiment
# 
#         outfile.write('\n>>XVset : {} : {} : {}<<\n'.format(experiment,XV_nb,factor))
#         print('\n>>XVset : {} : {} : {}<<\n'.format(experiment,XV_nb,factor))
# 
#         save_dir = '/home/rpizarro/noise/figs/performance/mc/{}/{}/{}/{}/{}/roc'.format(experiment,XV_nb,XV_set,classes,factor)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir,exist_ok=True)
# 
#         # This is prior to uncertainty measures... will skip for now
#         # df_clean = gen_df_clean(df)
#         # comp_stats(df_clean)
#         # plot_roc_conf_threshold(save_dir,df_clean)
# 
#         if uncertainty:
#             # This is for uncertainty measures
#             # compare_unc_by_threshold(outfile,save_dir,df,unc_thresholds=[0.5,0.1,0.01])
#             compare_threshold_by_unc(outfile,save_dir,df,unc_cols = ['MI','entropy','sample_variance'])
#         
#         df_base = gen_df_clean(df)
#         plt.plot(df_base.fpr,df_base.tpr,'{}-'.format(colors[idx]),label=factor)
#         
#         summary_stats = comp_stats(df_base)
#         df_stats = append_stats(df_stats,factor,summary_stats)
# 
#         outfile.close()
# 
#     # save stats to csv
#     # fn = os.path.join(pred_dir_root,'summary_stats.arch_size.csv')
#     fn = os.path.join(pred_dir_root,'summary_stats.nb_sample_factors.csv')
#     # fn = os.path.join(pred_dir_root,'summary_stats.batch_insta_norm.csv')
#     # fn = os.path.join(pred_dir_root,'summary_stats.augmented_comparison.csv')
#     print('Saving to : {}'.format(fn))
#     df_stats.to_csv(fn)
# 
#     plt.axis([0.0, 0.5, 0.2, 1.0])
#     plt.tick_params(labelsize=15)
#     plt.legend(loc='lower right',fontsize=17)
#     plt.xlabel('false positive rate',fontsize=17)
#     plt.ylabel('true positive rate',fontsize=17)
#     plt.tight_layout()
#     plt.grid(True)
# 
#     # fn = os.path.join(Path(save_dir).parents[1],'roc.augmented_comparison.png')
#     # fn = os.path.join(Path(save_dir).parents[1],'roc.batch_insta_norm.png')
#     # fn = os.path.join(Path(save_dir).parents[1],'roc.arch_size.png')
#     fn = os.path.join(Path(save_dir).parents[1],'roc.nb_sample_factors.png')
#     # fn = os.path.join(Path(save_dir).parents[1],'roc.LR_comparison.png')
#     print('Saving to : {}'.format(fn))
#     plt.savefig(fn)
#     plt.close(fig_nb)


def check_dirs(epoch_dirs_full,XV_set):
    epoch_dirs = []
    for ed in epoch_dirs_full:
        fn = os.path.join(ed,'{}_stats_summarized.csv'.format(XV_set))
        print(fn)
        if os.path.exists(fn):
            epoch_dirs += [ed]
    print('We found the following dirs to be analyzed\n{}'.format(epoch_dirs))
    return epoch_dirs

def mc_plot_roc_epochs(factors,experiment,XV_nb,XV_set,classes,sub_experiment,uncertainty=True):
    pred_dir = '/trials/data/rpizarro/noise/prediction/'
    figs_dir = '/trials/data/rpizarro/noise/figs/performance/mc/'
    # epochs = ['ep0090','ep0250','ep0270']
    for factor in factors:
        mc_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor)
        # mc_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor,'epoch_transience')
        # epochs_dir = glob.glob(os.path.join(mc_dir,sub_experiment))
        # epochs_dir = check_dirs(epochs_dir,XV_set)
        # epochs_dir = glob.glob(os.path.join(mc_dir,'002-ramp_clean_050_to_098_ep0500','epoch_transience','ep*'))
        fig_nb = 1001
        plt.figure(fig_nb,figsize=(8,6))
        # ep_clr = ['k:','k--','k--','k--','r-','r-','r-','g-','b-','b-','b-','m-']
        ep_clr = ['k:','k--','r:','r--','g:','g--','b:','b--']
        # labels = ['S3_base_random','S3_base_initialized','S3_base_initialized N2','S3_base_initialized N3','S3_dr_ER_005','S3_dr_ER_005 N2','S3_dr_ER_005 N3','S3_dr_ER_020','S3_dr_ER_100','S3_dr_ER_100 N2','S3_dr_ER_100 N3','S3_dr_ER_200']
        labels = ['000','001','010','011','100','101','110','111']
        # labels = ['baseline (random)','baseline (initialized)','baseline (initialized) N_2',r'data_ramp, $E_R$=5',r'data_ramp, $E_R$=5 N_2',r'data_ramp, $E_R$=20',r'data_ramp $E_R$=100',r'data_ramp $E_R$=100 N_2']
        df_stats = pd.DataFrame(columns=['Factor','nb_epoch','AUC','thresh_maxJ','accuracy','sensitivity','specificity','tp','fn','fp','tn'])
        # ed_list = sorted(epochs_dir)
        # We want to display random experiment first
        # if len(ed_list)>1:
        #     ed_list[0],ed_list[1] = ed_list[1],ed_list[0]
        xJ = []
        yJ = []
        for idx,epoch_dir in enumerate(sub_experiment):
            ed = os.path.join(mc_dir,epoch_dir)
            ed = check_dirs([ed],XV_set)
            if not ed:
                # returned an empty dir list
                continue
            else:
                ed = ed[0]

            ep = os.path.basename(ed)
            fn = os.path.join(ed,'{}.mc_plot_roc.out'.format(XV_set))
            outfile = open(fn, 'w')
            fn = os.path.join(ed,'{}_stats_summarized.csv'.format(XV_set))
            if os.path.exists(fn): # in epochs:
                pass
                # print('Will analyze : {}'.format(ep))
            else:
                print('Does not exist: {}'.format(fn))
                print('Will skip {}'.format(ep))
                continue
            df = pd.read_csv(fn,index_col=0)

            outfile.write('\n>>XVset : {} : {} : {} : {}<<\n'.format(experiment,XV_nb,factor,ep))
            print('\n>>XVset : {} : {} : {} : {}<<\n'.format(experiment,XV_nb,factor,ep))
            df.pred_dir = ed
            df.XV_nb = XV_nb
            df.experiment = experiment
            df.nb_epoch = ep
            XVset_length = df.shape[0]

            save_dir = os.path.join(figs_dir,experiment,XV_nb,XV_set,classes,factor,ep)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            print('We will save figures to : {}'.format(save_dir)) 
            if uncertainty:
                # This is for uncertainty measures
                # compare_unc_by_threshold(outfile,save_dir,df,unc_thresholds=[0.5,0.1,0.01])
                compare_threshold_by_unc(outfile,save_dir,df,unc_cols = ['MI','entropy','sample_variance'])
            df_base = gen_df_clean(df)
            plt.plot(df_base.fpr,df_base.tpr,ep_clr[idx],label=labels[idx])
            # AUC,threshold,accuracy,sensitivity,specificity 
            summary_stats = comp_stats(df_base,set_sz=XVset_length,J='J')
            xJ += [1-summary_stats[4]]
            yJ += [summary_stats[3]]
            plt.plot(1-summary_stats[4],summary_stats[3],'{}*'.format(ep_clr[idx][0]),markersize=10)
            df_stats = append_stats(df_stats,factor,summary_stats,nb_epoch=ep)

        plt.plot([0.0,1.0],[0.0,1.0],'c-',linewidth=2.0,label='naive')
        BBK_sens = 0.91
        BBK_spec = 0.84
        # plt.plot(1-BBK_spec,BBK_sens,'cX',markersize=10,label='Biobank')

        plt.axis([-0.05, 1.05, -0.05, 1.05])
        plt.tick_params(labelsize=15)
        legend = plt.legend(loc='lower right',fontsize=17,framealpha=1.0)
        plt.xlabel('false positive rate',fontsize=17)
        plt.ylabel('true positive rate',fontsize=17)
        # plt.title(ep,fontsize=20)
        plt.tight_layout()
        plt.grid(True)

        fn = os.path.join(os.path.dirname(save_dir),'roc.comparison_experiments_N2.png')
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)
        
        legend.remove()
        fn = os.path.join(os.path.dirname(save_dir),'roc.comparison_experiments_N2.nolegend.png')
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)
        
        ax = plt.gca()
        # stage 2
        # plt.axis([-0.005, 0.055, 0.945, 1.005])
        # stage 3 ramping
        plt.axis([-0.01, 0.21, 0.75, 1.01])
        ax.xaxis.tick_top()
        # plt.xticks([0.05,0.1],(0.05,0.1),fontsize=40)
        plt.xticks([0.0,0.1,0.2],('0.0','0.1','0.2'),fontsize=40)
        plt.yticks([0.8,0.9,1.0],('0.8','0.9','1.0'),fontsize=40)
        for (x,y,clr) in zip(xJ,yJ,ep_clr):
            print(x,y,clr)
            plt.plot(x,y,'{}*'.format(clr),markersize=40)
        # plt.plot(1-BBK_spec,BBK_sens,'cX',markersize=40,label='Biobank')
        plt.xlabel(None)
        plt.ylabel(None)
        plt.tight_layout()
        fn = os.path.join(os.path.dirname(save_dir),'roc.comparison_experiments_N2.zoom.png')
        outfile.write('Saving to : {}\n'.format(fn))
        print('Saving to : {}'.format(fn))
        plt.savefig(fn)
        
        plt.close(fig_nb)
        outfile.close()

        # save stats to csv
        fn = os.path.join(mc_dir,'summary_stats.comparison_experiments_N2.csv')
        print('Saving to : {}'.format(fn))
        df_stats.to_csv(fn)


view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())



start_time = time.time()

# This script will execute after running noise.mc_summarize_prediction.py
# python noise.mc_gen_conf_mtrx.py rap_NN007_100ep_CLR test nb_classes_02 True
XV_set = sys.argv[1]
uncertainty = sys.argv[2]
# sub_experiment = sys.argv[3]

experiment = 'rap_NN008_multiple_artifact/clean_percent_098' # sys.argv[1]
# experiment = 'rap_NN008_multiple_artifact/clean_percent_stg1' # sys.argv[1]
XV_nb = 'XV0' # sys.argv[2]
classes = 'nb_classes_02' # sys.argv[4]
factors = ['nb_samples_factor_01.00']

sub_experiment = [
        '001-random_constant_clean_098_ep0050',
        '001-initialized_constant_clean_098_ep0050',
        '001-initialized_constant_clean_098_ep0050_N2',
        '001-initialized_constant_clean_098_ep0050_N3',
        '002-E1ramp_005_data_800_to_20000_ep0053',
        '002-E1ramp_005_data_800_to_20000_ep0053_N2',
        '002-E1ramp_005_data_800_to_20000_ep0053_N3',
        '002-E1ramp_020_data_800_to_20000_ep0060',
        '002-E1ramp_100_data_800_to_20000_ep0100',
        '002-E1ramp_100_data_800_to_20000_ep0100_N2',
        '002-E1ramp_100_data_800_to_20000_ep0100_N3',
        ]
sub_experiment = ['001-CLR-trg12_NN8mod_priming_ep500']
sub_experiment = ['001-random_constant_clean_098_ep0050_focal']
sub_experiment = ['002-E1ramp_100_data_800_to_20000_ep0100']


sub_experiment = ['S3_dr_ER_100_mc000',
        'S3_dr_ER_100_mc001',
        'S3_dr_ER_100_mc002',
        'S3_dr_ER_100_mc003',
        'S3_dr_ER_100_mc004',
        'S3_dr_ER_100_mc005',
        'S3_dr_ER_100_mc006',
        'S3_dr_ER_100_mc007']


sub_experiment = ['001-random_constant_clean_stg1_ep0200_NN7_combat']

sub_experiment = [
        '001-random_constant_clean_098_ep0050_focal',
        '001-random_constant_clean_098_ep0050_focal_by10'
        ]

sub_experiment = ['S3_initi_050_mc000',
        'S3_initi_050_mc001',
        'S3_initi_050_mc002',
        'S3_initi_050_mc003',
        'S3_initi_050_mc004',
        'S3_initi_050_mc005',
        'S3_initi_050_mc006',
        'S3_initi_050_mc007']

# mc_plot_roc_factor(factors,experiment,XV_nb,XV_set,classes,uncertainty=='True')
sub_experiment = ['001-initialized_constant_clean_098_ep0050_flip']

mc_plot_roc_epochs(factors,experiment,XV_nb,XV_set,classes,sub_experiment,uncertainty=='True')


elapsed_time = time.time() - start_time
print('We analyzed all factors in : {}'.format(factors))
print('Time it took to compute ROC curves : {0:0.2f} seconds'.format(elapsed_time))



