import os,sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_lr(ep,epochs=100):
    lr_max=1
    if epochs==100:
        lr_max=0
    lr = np.logspace(-12, lr_max, epochs)
    return lr[ep]

def get_parms(fn):
    with open(fn) as json_data:
        p = json.load(json_data)
    parms_dict = json.loads(p)
    return parms_dict


def get_history(experiment):
    hist_dir = '/data/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
    hist_files = glob.glob(os.path.join(hist_dir,'history*parms.json'))
    if len(hist_files)>1:
        hist_files.sort()
        # initializing dictionary keys
        parms_dict = get_parms(hist_files[0])
        print(parms_dict)
        for fn in hist_files[1:]:
            curr_dict = get_parms(fn)
            for metric in parms_dict.keys():
                parms_dict[metric] += curr_dict[metric]
        df = pd.DataFrame.from_dict(parms_dict)
    else:
        manual_dir = '/home/rpizarro/noise/weights/{}'.format(experiment)
        fn = os.path.join(manual_dir,'training_loss_by_epoch.txt')
        df = pd.read_csv(fn,header=None).rename(columns={0:'loss'})

    return df


def get_dloss_ave(df,lr_cutoff,window):
    epochs=len(df.loss)
    df['lr'] = [get_lr(ep,epochs=epochs) for ep in df.index]
    df = df.loc[df['lr'] < lr_cutoff]
    df['loss_shift'] = df.loss.shift(1)
    df['dloss'] = df.loss-df.loss_shift
    df['dloss_ave'] = df['dloss'].rolling(window).mean()
    print(df)
    return df

def plot_4_figs(df,window):
    fig, ax = plt.subplots()

    # plot differential loss ave by learning rate
    plt.plot(df.lr,df.dloss_ave)
    epochs=len(df.loss)
    min_idx = df['dloss_ave'].idxmin()
    minx=df.lr[min_idx]
    miny=df.dloss_ave[min_idx]
    plt.plot(minx,miny,'r*',markersize='12')
    ax.annotate('({0:1.2e},{1:0.2f})'.format(minx,miny),xy=(minx, miny), xytext=(minx*100, miny*0.9),
                        arrowprops=dict(arrowstyle = "->"))
    plt.xscale('log')
    plt.grid('True')
    plt.ylabel('d/loss moving average {}'.format(window))
    plt.xlabel('learning rate (log scale)')

    fn = os.path.join(save_dir,'dloss-ave_x_lr.{}ep.png'.format(epochs))
    print('Saving differential loss ave by learning rate plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()

    # plot learning rate by epoch
    plt.plot(df.index+1,df.lr)
    plt.grid('True')
    plt.ylabel('learning rate',fontsize=15)
    plt.xlabel('epoch',fontsize=15)
    plt.tick_params(labelsize=12)

    fn = os.path.join(save_dir,'lr_x_epoch.{}ep.png'.format(epochs))
    print('Saving learning rate by epoch plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()

    # plot traning loss by learning rate
    plt.plot(df.lr,df.loss)
    plt.xscale('log')
    plt.grid('True')
    plt.ylabel('loss')
    plt.xlabel('learning rate (log scale)')

    fn = os.path.join(save_dir,'loss_x_lr.{}ep.png'.format(epochs))
    print('Saving loss by learning rate plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()

    # plot differential loss by learning rate
    plt.plot(df.lr,df.dloss)
    plt.xscale('log')
    plt.grid('True')
    plt.ylabel('d/loss')
    plt.xlabel('learning rate (log scale)')

    fn = os.path.join(save_dir,'dloss_x_lr.{}ep.png'.format(epochs))
    print('Saving differential loss by learning rate plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()

def plot_single_fig_2by1(df,window):

    f = plt.figure(1,figsize=(10,10))
    # plot traning loss by learning rate
    ax = plt.subplot(211)
    plt.plot(df.lr[window:],df.loss[window:])
    plt.xscale('log')
    plt.grid('True')
    plt.ylabel('loss',fontsize=25)
    plt.tick_params(labelsize=15)
    ax.set_xticklabels([])
    # plt.xlabel('learning rate (log scale)')

    # plot differential loss ave by learning rate
    ax2=plt.subplot(212)
    plt.plot(df.lr,df.dloss_ave)
    min_idx = df['dloss_ave'].idxmin()
    minx=df.lr[min_idx]
    miny=df.dloss_ave[min_idx]
    plt.plot(minx,miny,'r*',markersize='12')
    ax2.annotate('({0:1.2e},{1:0.2f})'.format(minx,miny),xy=(minx, miny), xytext=(minx*100, miny*0.9),fontsize=15,
                        arrowprops=dict(arrowstyle = "->"))
    plt.xscale('log')
    plt.grid('True')
    plt.ylabel('d/loss moving average {}'.format(window),fontsize=25)
    plt.xlabel('learning rate (log scale)',fontsize=25)
    plt.tick_params(labelsize=15)

    fn = os.path.join(save_dir,'lr_loss_dloss.{}ep.png'.format(len(df.loss)))
    print('Saving differential loss ave by learning rate plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()



# python noise.plot_loss_lr.py rap_NN007_bin_append window 
experiment = sys.argv[1]
window = int(sys.argv[2])
save_dir = '/home/rpizarro/noise/figs/tune_lr/{}'.format(experiment)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = get_history(experiment)

lr_cutoff = 11
# window = 20
df = get_dloss_ave(df,lr_cutoff,window)

plot_4_figs(df,window)

plot_single_fig_2by1(df,window)



