import pandas as pd
import os,sys
import matplotlib.pyplot as plt



def append_new_metrics(df,sub_experiment):
    # some housekeeping
    df = df.drop(['XVset'], axis=1)
    df = df.rename(columns={"retained": "retained_%"})
    df['accuracy'] = 100.0 * df['accuracy']
    df['sensitivity'] = 100.0 * df['sensitivity']
    df['specificity'] = 100.0 * df['specificity']
    # df.loc[df['eta_thresh']==1.0,['unc_metric']]='none'
    df['sub_experiment'] = sub_experiment
    # df = df.drop_duplicates()

    # let's append
    df['retained_nb'] = df['tp']+df['fn']+df['fp']+df['tn']
    set_sz = df['retained_nb'][0]
    df['sent_nb'] = set_sz - df['retained_nb']
    df['class_pos'] = df['tp'] + df['fp']
    df['DFFMR_nb'] = df['sent_nb'] + df['class_pos']
    df['DFFMR_%'] = 100.0 * df['DFFMR_nb'] / set_sz
    df['UDM_%'] = 100.0 * df['fn'] / set_sz
    df['P.prediction_%'] = 100.0 * df['tp'] / (df['tp'] + df['fp'])
    df['N.prediction_%'] = 100.0 * df['tn'] / (df['tn'] + df['fn'])
    
    cols = list(df)
    cols = [cols[i] for i in (2,3,4,5,6,7,17,18,19,20)]
    decimals = pd.Series([1,3,2,1,1,1,2,2,1,1], index=cols)
    df = df.round(decimals)

    return df

def add_BBK(plt,th,markersize=10):
    plt.plot(2*[th[0]],[0,1],'m--',linewidth=markersize*0.75/10)
    plt.plot([0,100],2*[th[1]],'m--',linewidth=markersize*0.75/10)
    plt.plot(th[0],th[1],'mX',markersize=2*markersize,label='ML')
    return plt

def plot_by_markersize(fig_nb,df_all,sub_ex_list,th,markersize=10):
    unc_metrics = ['MI','entropy','sample_variance']
    plt.figure(fig_nb,figsize=(8,6))
    clr_marker = ['ko','k+','kx','k*','r+','rx','r*','g+','b+','bx','b*','m+']
    labels = ['random','initialized','initialized N2','initialized N3','ER=5','ER=5 N2','ER=5 N3','ER=20','ER=100','ER=100 N2','ER=100 N3','ER=200']
    for index,sub_experiment in enumerate(sub_ex_list):
        df_sub_ex = df_all[df_all['sub_experiment']==sub_experiment].reset_index(drop=True)
        for unc_idx,unc in enumerate(unc_metrics):
            df_unc = df_sub_ex[df_sub_ex['unc_metric']==unc]
            # for unc_idx,row in df_unc.iterrows():
            plt.plot(df_unc['DFFMR_%'],df_unc['UDM_%'],'{}'.format(clr_marker[index]),markersize=markersize,linewidth=0.1,label=labels[index] if unc_idx == 0 else '')
    return plt

def plot_DFFMR_UDM(df_all,sub_ex_list,th=(17.29,0.15)):

    fig_nb = 1001
    plt = plot_by_markersize(fig_nb,df_all,sub_ex_list,th,markersize=10)
    plt.axis([0,100,0,0.5])
    plt.tick_params(labelsize=15)
    plt.ylabel('Unusable datasets missed (%)',fontsize=15)
    plt.xlabel('Datasets flagged for manual review (%)',fontsize=15)
    legend = plt.legend(loc='upper right')
    fig_fn = '/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.DFFMR_UDM_noBBK.png'
    plt.tight_layout()
    print('Saving to : {}'.format(fig_fn))
    plt.savefig(fig_fn)
    
    legend.remove()
    plt = add_BBK(plt,th,markersize=10)
    legend = plt.legend(loc='upper right')
    fig_fn = '/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.DFFMR_UDM.png'
    plt.tight_layout()
    print('Saving to : {}'.format(fig_fn))
    plt.savefig(fig_fn)

    plt.close(fig_nb)

    fig_nb = 1001
    plt = plot_by_markersize(fig_nb,df_all,sub_ex_list,th,markersize=30)
    plt = add_BBK(plt,th,markersize=30)
    plt.axis([5, 23, 0.05, 0.2])
    plt.xticks([5,10,15,20],('5','10','15','20'),fontsize=30)
    plt.yticks([0.05,0.1,0.15,0.2],('0.05','0.10','0.15','0.20'),fontsize=30)
    fig_fn = '/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.DFFMR_UDM_zoom.png'
    plt.tight_layout()
    print('Saving to : {}'.format(fig_fn))
    plt.savefig(fig_fn)
    plt.close(fig_nb)



# This script will execute after running noise.mc_summarize_prediction.py
# python noise.mc_gen_conf_mtrx.py rap_NN007_100ep_CLR test nb_classes_02 True
XV_set = 'valid'
# sub_experiment = sys.argv[3]

experiment = 'rap_NN008_multiple_artifact/clean_percent_098' # sys.argv[1]
XV_nb = 'XV0' # sys.argv[2]
classes = 'nb_classes_02' # sys.argv[4]
factor = 'nb_samples_factor_01.00'

sub_experiment_list = [
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

# sub_experiment_list = ['002-E1ramp_100_data_800_to_20000_ep0100']


pred_dir = '/data/rpizarro/noise/prediction/'
mc_dir = os.path.join(pred_dir,experiment,XV_nb,classes,factor)

# state of the art to beat
DFFMR_threshold = 17.29
UDM_threshold = 0.15

cols = ['sub_experiment','eta_thresh', 'unc_metric', 'retained_%', 'AUC', 'thresh_maxJ', 'accuracy', 'sensitivity', 'specificity', 'tp', 'fn', 'fp', 'tn', 'retained_nb', 'sent_nb', 'class_pos', 'DFFMR_nb', 'DFFMR_%', 'UDM_%', 'P.prediction_%', 'N.prediction_%']

df_out = pd.DataFrame(columns=cols)
df_all = pd.DataFrame(columns=cols)

for sub_experiment in sub_experiment_list:
    print('\n==Working on sub_experiment : {}=='.format(sub_experiment))
    fn = os.path.join(mc_dir,sub_experiment,'summary_stats.threshold_by_unc.csv')
    df = pd.read_csv(fn,index_col=0)
    df = append_new_metrics(df,sub_experiment)
    df_all = df_all.append(df,ignore_index=True)
    # print(df)
    df_outperform = df[(df['DFFMR_%'] < DFFMR_threshold) & (df['UDM_%'] < UDM_threshold)]
    if df_outperform.empty:
        print('We could not find a threshold that outperformed Biobank...')
    else:
        print('We found the following eta threshold that outperforms Biobank!!!!\n')
        print(df_outperform)
        df_out = df_out.append(df_outperform,ignore_index=True)


# remove the duplicate entries with no threshold or eta<1.0
df_out.loc[df_out['eta_thresh']==1.0,['unc_metric']]='none'
df_out = df_out.drop_duplicates()
df_out = df_out[cols]

fn = os.path.join(mc_dir,'outperform_BBK_stats.threshold_by_unc.csv')
print('Saving the threshold values that outperform Biobank to {}'.format(fn))
df_out.to_csv(fn)
print(df_out)

df_all = df_all[cols]
fn = os.path.join(mc_dir,'appended_BBK_stats.threshold_by_unc.csv')
print('Saving the all threshold values with Biobank stats to {}'.format(fn))
df_all.to_csv(fn)
print(df_all)

plot_DFFMR_UDM(df_all,sub_experiment_list,th=(DFFMR_threshold,UDM_threshold))




