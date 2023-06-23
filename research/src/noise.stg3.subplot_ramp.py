import pandas as pd
import glob
import os,sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def get_size_cleanliness(fn):
    df = pd.read_csv(fn,index_col=0)
    # print(df)
    nb_clean = df['clean'].sum()
    nb_artifacts = df['artifact'].sum()
    # print(df[['clean','artifact']].sum())
    # print(nb_clean,nb_artifacts)
    clean_percent = (1.0*nb_clean) / (nb_clean + nb_artifacts)
    # print(clean_percent)
    return int(nb_clean),int(nb_artifacts),clean_percent

def plot_ramp(df,df_time):
    fig_nb=1234
    plt.figure(fig_nb,figsize=(6,12))
    plt.subplot(311)
    plt.plot(df['epoch'],100*df['clean_percent'])
    plt.ylabel('Clean percentage (%)',fontsize=20)
    plt.ylim([0,100])
    plt.xlim([0,500])
    plt.text(20,80,'(a)',fontsize=20)
    plt.tight_layout()

    plt.subplot(312)
    plt.plot(df['epoch'],df['nb_MRI_volumes'])
    plt.ylabel('Dataset size (# MRI vols)',fontsize=20)
    plt.text(20,17500,'(b)',fontsize=20)
    plt.ylim([0,21000])
    plt.xlim([0,500])
    plt.tight_layout()

    plt.subplot(313)
    plt.plot(df['epoch'],df_time['seconds']/60.0)
    plt.xlabel('Epoch number',fontsize=20)
    plt.ylabel('Training time (min)',fontsize=20)
    plt.text(20,140,'(c)',fontsize=20)
    plt.ylim([0,160])
    plt.xlim([0,500])
    plt.tight_layout()
    
    save_dir = '/home/rpizarro/noise/figs/002-ramp_clean_050_to_098_ep050'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'ramp_by_epoch_abc.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def plot_clean_percentage_ramp(df,experiment):
    ep_max = int(experiment.split('ep')[1])
    fig_nb=1234
    plt.figure(fig_nb,figsize=(5,8))
    
    plt.subplot(211)
    plt.plot(df['epoch'],100*df['clean_percent'])
    plt.ylabel('Clean percentage (%)',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim([50,100])
    plt.xlim([0,ep_max])
    plt.text(0.04*ep_max,90,'(a)',fontsize=20)
    plt.tight_layout()
    
    plt.subplot(212)
    plt.plot(df['epoch'],df['nb_MRI_volumes'])
    plt.ylabel('Dataset size (# MRI vols)',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim([0,21000])
    plt.xlim([0,ep_max])
    plt.text(0.04*ep_max,17500,'(b)',fontsize=20)
    plt.xlabel('Epoch number',fontsize=20)
    plt.tight_layout()
    
    save_dir = os.path.join('/home/rpizarro/noise/figs/',experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'clean_percentage_ramp.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def plot_future_ramps():
    fig_nb=1234
    plt.figure(fig_nb,figsize=(6,8))
    
    plt.subplot(311)
    plt.plot([0,5,53],[800,20000,20000])
    plt.tick_params(labelsize=15)
    plt.ylim([0,21000])
    plt.xlim([0,53])
    plt.text(0.04*53,17500,'(a)',fontsize=20)
    plt.tight_layout()
    
    plt.subplot(312)
    plt.plot([0,20,60],[800,20000,20000])
    plt.tick_params(labelsize=15)
    plt.ylabel('Dataset size (# MRI vols)',fontsize=20)
    plt.ylim([0,21000])
    plt.xlim([0,60])
    plt.text(0.04*60,17500,'(b)',fontsize=20)
    plt.tight_layout()
    
    plt.subplot(313)
    plt.plot([0,100],[800,20000])
    plt.tick_params(labelsize=15)
    plt.ylim([0,21000])
    plt.xlim([0,100])
    plt.xlabel('Epoch number',fontsize=20)
    plt.text(0.04*100,17500,'(c)',fontsize=20)
    plt.tight_layout()
    
    save_dir = os.path.join('/home/rpizarro/noise/figs/','future_experiments')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'data_ramp.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def plot_clean_ramp_window(df,experiment,title_str='Title Figure',lb1='(a)',lb2='(b)',label=True):
    ep_max = int(experiment.split('ep')[1])
    fig_nb=1234

    fig, ax = plt.subplots(figsize=(8.5,4))
    # fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()

    p1, = ax.plot(df['epoch'],100*df['clean_percent'], "b-", label="clean %")
    p2, = twin1.plot(df['epoch'],df['nb_MRI_volumes'], "k-", label="# MRI")

    ax.set_xlim(0, ep_max)
    ax.set_ylim(50, 100)
    ax.tick_params(labelsize=20)
    twin1.set_ylim(0, 21000)
    twin1.tick_params(labelsize=20)

    ax.set_xlabel("training epochs",fontsize=25)
    ax.set_ylabel("clean percent (%)",fontsize=25)
    twin1.set_ylabel("train size (# MRI)",fontsize=25)

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    tight_layout()
    ax.grid(True)
    ax.legend(handles=[p1, p2],loc='upper left',fontsize=20)
    ax.text(0.6*ep_max,50+0.8*50,"(a)",fontsize=25)
    save_dir = os.path.join('/home/rpizarro/noise/figs/',experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'data_ramp_window.png')
    if not label:
        fn = os.path.join(save_dir,'data_ramp_label_off_window.png')
    print('Saving to : {}'.format(fn))
    savefig(fn,dpi=72)

    ax.set_xticklabels([])
    twin1.set_xticklabels([])
    ax.set_xlabel(None)
    tight_layout()
    fn = os.path.join(save_dir,'data_ramp_window_noX.png')
    print('Saving to : {}'.format(fn))
    savefig(fn,dpi=72)

    # plt.close(fig_nb)



def plot_data_ramp_window(df,experiment,title_str='Title Figure',lb1='(a)',lb2='(b)',xsize=5.5,label='All'):
    ep_max = int(experiment.split('ep')[1])
    fig_nb=1234

    fig, ax = plt.subplots(figsize=(xsize,6))
    # fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()

    p1, = ax.plot(df['epoch'],df['nb_MRI_volumes'], "k-", label="# MRI")
    p2, = twin1.plot(df['epoch'],100*df['clean_percent'], "b-", label="clean %")
    

    ax.set_xlim(0, ep_max)
    ax.set_ylim(0, 21000)
    ax.tick_params(labelsize=20)
    twin1.set_ylim(50, 100)
    twin1.tick_params(labelsize=20)

    ax.set_xlabel("training epochs",fontsize=25)
    if 'Right' not in label:
        ax.set_ylabel("train size (# MRI)",fontsize=25)
    else:
        ax.set_yticklabels([])
    if 'Left' not in label:
        twin1.set_ylabel("clean percentage (%)",fontsize=25)
    else:
        twin1.set_yticklabels([])

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    plt.text(0.6*ep_max,70,lb2,fontsize=25)
    plt.title('$\it{}$'.format(title_str),fontsize=25)
    tight_layout()
    
    if 'Left' not in label:
        ax.legend(handles=[p1, p2],loc='lower right',fontsize=20)

    save_dir = os.path.join('/data/rpizarro/noise/figs/',experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'data_ramp_window_{}.png'.format(label))
    print('Saving to : {}'.format(fn))
    savefig(fn,dpi=72)
    # plt.close()



def plot_data_ramp(df,experiment,title_str='Title Figure',lb1='(a)',lb2='(b)',label=True):
    ep_max = int(experiment.split('ep')[1])
    fig_nb=1234
    fig = plt.figure(fig_nb,figsize=(5,8))
    
    ax = plt.subplot(211)
    plt.plot(df['epoch'],df['nb_MRI_volumes'])
    if label:
        plt.ylabel('Dataset size (# MRI vols)',fontsize=20)
    else:
        ax.set_yticklabels([])
    plt.tick_params(labelsize=15)
    ax.set_xticklabels([])
    plt.ylim([0,21000])
    plt.xlim([0,ep_max])
    plt.text(0.8*ep_max,5000,lb1,fontsize=20)
    plt.title('$\it{}$'.format(title_str),fontsize=25)
    plt.tight_layout()
    
    ax = plt.subplot(212)
    plt.plot(df['epoch'],100*df['clean_percent'])
    if label:
        plt.ylabel('Clean percentage (%)',fontsize=20)
    else:
        ax.set_yticklabels([])
    plt.tick_params(labelsize=15)
    plt.ylim([50,100])
    plt.xlim([0,ep_max])
    plt.text(0.8*ep_max,60,lb2,fontsize=20)
    plt.xlabel('Epoch number',fontsize=20)
    plt.tight_layout()

    fig.subplots_adjust(hspace=0.1)
    
    save_dir = os.path.join('/home/rpizarro/noise/figs/',experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fn = os.path.join(save_dir,'data_ramp.png')
    if not label:
        fn = os.path.join(save_dir,'data_ramp_label_off.png')
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)

def get_epoch(fn):
    epoch = fn.replace('train.art123.ep','').replace('.csv','')
    return int(epoch)

# dr_050 width = 231: 5.5*231/311 = 4.1 (5.5)
# dr_060 width = 196: 5.5*196/311 = 3.5 (4.7)
# dr_100 width = 396: 5.5*396/311 = 7.0 (9.4)

experiment = sys.argv[1]
title_str = sys.argv[2]
lb1 = sys.argv[3]
lb2 = sys.argv[4]
xsize = float(sys.argv[5])

weights_dir = os.path.join('/data/rpizarro/datasets/shared/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/',experiment)

cols = ['epoch','nb_clean','nb_artifact','nb_MRI_volumes','clean_percent']
df = pd.DataFrame(columns=cols)

train_files = glob.glob(os.path.join(weights_dir,'train.art123.ep*.csv'))

for train_fn in sorted(train_files):
    # train_fn = os.path.join(weights_dir,'train.art123.ep{0:04d}.csv'.format(ep))
    nb_clean,nb_artifacts,clean_percent = get_size_cleanliness(train_fn)
    ep = get_epoch(os.path.basename(train_fn))
    row = pd.DataFrame([[ep,nb_clean,nb_artifacts,nb_clean+nb_artifacts,clean_percent]],columns=cols)
    df = df.append(row,ignore_index=True)


ep_max = int(experiment.split('ep')[1])
row['epoch'] = ep_max
# row = pd.DataFrame([[500,20462,417,20879,0.98]],columns=cols)
df = df.append(row,ignore_index=True)

print(df)

# plot_clean_percentage_ramp(df,experiment)
if '002-ramp_clean_050_to_098_ep0500' in experiment:
    plot_clean_ramp_window(df,experiment,title_str,lb1,lb2,label=True)
else:
    plot_data_ramp_window(df,experiment,title_str,lb1,lb2,xsize,label='All')
    plot_data_ramp_window(df,experiment,title_str,lb1,lb2,xsize,label='Left')
    plot_data_ramp_window(df,experiment,title_str,lb1,lb2,xsize,label='Right')
    plot_data_ramp_window(df,experiment,title_str,lb1,lb2,xsize,label='LeftRight')

# plot_data_ramp(df,experiment,title_str,lb1,lb2,label=True)
# plot_data_ramp(df,experiment,title_str,lb1,lb2,label=False)
# plot_future_ramps()
# append time but it is identical to datasize

sys.exit()


'''
Omit for now

weights_log_dir = '/home/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_098/XV0'
epoch_time_fn = os.path.join(weights_log_dir,'epoch_time.csv')
df_time = pd.read_csv(epoch_time_fn)
print(df_time)

time_min_500ep = df_time['seconds'].sum()/60.0
print('The time it took for 500 epochs stage 3 ramp experiment : {0:0.3f} hours, or {1:0.3f} days'.format(time_min_500ep/60.0,time_min_500ep/60.0/24))

plot_ramp(df,df_time)
'''
'''
    axL = subplot(1,1,1)
    plot(df['epoch'],df['nb_MRI_volumes'])
    ylabel('Dataset size (# MRI vols)',fontsize=20)
    axL.tick_params(labelsize=15)
    # ylim([0,21000])
    xlim([0,ep_max])
    title('$\it{}$'.format(title_str),fontsize=25)

    axR = plt.subplot(1,1,1, sharex=axL, frameon=False)
    axR.yaxis.tick_right()
    axR.yaxis.set_label_position("right")
    plot(df['epoch'],100*df['clean_percent'])
    ylabel('Clean percentage (%)',fontsize=20)
    axR.tick_params(labelsize=15)
    # ylim([50,100])
    # tight_layout()
    
    # fig.subplots_adjust(hspace=0.1)
    
'''

