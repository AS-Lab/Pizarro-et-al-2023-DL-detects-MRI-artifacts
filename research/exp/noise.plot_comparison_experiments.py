import pandas as pd
import os,sys
import matplotlib.pyplot as plt

def plot_AUC(df):
    fig_nb = 1001
    plt.figure(fig_nb,figsize=(8,6))
    labels = ['random','initialized','initialized N2','ER=5','ER=5 N2','ER=20','ER=100','ER=100 N2']
    plt.bar(labels,df['AUC'])
    plt.xticks(rotation=45)
    plt.ylim([0.9,1.0])
    plt.ylabel('AUC')
    bar_fn = '/home/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.AUC.png'
    plt.tight_layout()
    print('Saving to : {}'.format(bar_fn))
    plt.savefig(bar_fn)
    plt.close(fig_nb)

def plot_sens_spec(df):
    fig_nb = 1001
    plt.figure(fig_nb,figsize=(8,6))
    clr_marker = ['ko','k+','kx','r+','rx','g+','b+','bx']
    labels = ['random','initialized','initialized N2','ER=5','ER=5 N2','ER=20','ER=100','ER=100 N2']
    for index, row in df.iterrows():
        plt.plot(row['sensitivity'],row['specificity'],clr_marker[index],markersize=10,label=labels[index])
    plt.ylim([0.8,1.0])
    plt.xlim([0.75,1.0])
    plt.ylabel('specificity')
    plt.xlabel('sensitivity')
    plt.legend(loc='lower right')
    sens_spec_fn = '/home/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.sens_spec.png'
    plt.tight_layout()
    print('Saving to : {}'.format(sens_spec_fn))
    plt.savefig(sens_spec_fn)
    plt.close(fig_nb)

def plot_fp_fn(df):
    fig_nb = 1001
    plt.figure(fig_nb,figsize=(8,6))
    clr_marker = ['ko','k+','kx','r+','rx','g+','b+','bx']
    labels = ['random','initialized','initialized N2','ER=5','ER=5 N2','ER=20','ER=100','ER=100 N2']
    for index, row in df.iterrows():
        plt.plot(row['fn'],row['fp'],clr_marker[index],markersize=10,label=labels[index])
    plt.ylim([0,1000])
    plt.xlim([0,35])
    plt.ylabel('fp (clean class. as artifact)')
    plt.xlabel('fn (artifact class. as clean)')
    plt.legend(loc='lower right')
    sens_spec_fn = '/home/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/comparison_experiments.fp_fn.png'
    plt.tight_layout()
    print('Saving to : {}'.format(sens_spec_fn))
    plt.savefig(sens_spec_fn)
    plt.close(fig_nb)

fn = '/home/rpizarro/noise/prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/summary_stats.comparison_experiments_N2.csv'

df = pd.read_csv(fn,index_col=0)

print(df)


plot_AUC(df)

plot_sens_spec(df)

plot_fp_fn(df)



