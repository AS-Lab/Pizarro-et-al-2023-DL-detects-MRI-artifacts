import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os,sys
import glob
import numpy as np

def get_parms(fn):
    with open(fn) as json_data:
        p = json.load(json_data)
    parms_dict = json.loads(p)
    return parms_dict



def plot_cp_parms(experiment,parms_dict,plt_key = {0:1,1:2,2:1,3:2},title='',lb1='(a)',lb2='(b)'):
    save_dir = '/home/rpizarro/noise/figs/performance'
    max_tloss = max(parms_dict['loss'])
    max_vloss = max(parms_dict['val_loss'])
    max_loss = max(max_tloss,max_vloss)
    print(max_tloss,max_vloss)
    # number of epochs
    E = len(parms_dict['loss'])
    # moving window size
    N = E//20
    if E < 10:
        N = 1
    legend_label = ['training metric','moving average','validation metric','moving average']
    fig_nb = 1001
    plt.figure(fig_nb,figsize=(8.5,4.5))
    for mi,m in enumerate(sorted(parms_dict.keys())):
        print(mi,m,plt_key[mi])
        ax = plt.subplot(111)
        # full and same have boundary effects
        mov_ave = np.convolve(parms_dict[m], np.ones(N)/N, mode='valid')
        # mov_ave_x = [N//2+xi for xi in range(E-N+1)]
        mov_ave_x = [N//2+xi for xi in range(len(mov_ave))]
        if 'acc' in m:
            continue
        if 'val' in m:
            print('plot is in red')
            if 'loss' in m:
                vloss = [1.0*p for p in parms_dict[m]]
                plt.plot(vloss,'r:',linewidth=0.5,label=None)
                plt.plot(mov_ave_x,1.0*mov_ave,'r',linewidth=2,label='validation')
        else:
            print('plot is in green')
            plt.plot(parms_dict[m],'g:',linewidth=0.5,label=None)
            plt.plot(mov_ave_x,mov_ave,'g',linewidth=2,label='training')
        if plt_key[mi]==1:
            plt.title('$\it{}$'.format(title),fontsize=15)
            pass
        if plt_key[mi]==2:
            # plt.title('Validation',fontsize=10)
            pass
        legend = plt.legend(fontsize=20,loc='upper left')
        plt.text(0.6*E,0.8*1.0,"(b)",fontsize=25)
        if mi < 2:
        # if mi < 2:
            plt.ylabel(m.replace('acc','accuracy').replace('mean_squared_error','mse'),fontsize=25)
        plt.xlabel('training epochs',fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        ymax = 1.0
        plt.ylim(0,ymax)
        plt.xlim(0,E)
        ax.tick_params(labelsize=20)
    fn = os.path.join(save_dir,'{}_performance_loss_2by1.png'.format(experiment))
    print('Saving performance plot to : {}'.format(fn))
    plt.savefig(fn)

    legend.remove()
    fn = os.path.join(save_dir,'{}_performance_loss_nolegend.png'.format(experiment))
    print('Saving performance plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)





def plot_parms(experiment,parms_dict,plt_key = {0:1,1:2,2:1,3:2},title='',lb1='(a)',lb2='(b)'):
    save_dir = '/trials/data/rpizarro/noise/figs/performance'
    max_tloss = max(parms_dict['loss'])
    max_vloss = max(parms_dict['val_loss'])
    max_loss = max(max_tloss,max_vloss)
    print(max_tloss,max_vloss)
    # number of epochs
    E = len(parms_dict['loss'])
    # moving window size
    N = E//10
    if E < 10:
        N = 1
    legend_label = ['training metric','training metric, MA','validation metric','validation metric, MA']
    legend_label = ['training metric','training metric, MA','validation metric','validation metric, MA']
    fig_nb = 1001
    plt.figure(fig_nb,figsize=(4,5))
    for mi,m in enumerate(sorted(parms_dict.keys())):
        print(mi,m,plt_key[mi])
        plot_nb = 210+plt_key[mi]
        ax = plt.subplot(plot_nb)
        # full and same have boundary effects
        mov_ave = np.convolve(parms_dict[m], np.ones(N)/N, mode='valid')
        # mov_ave_x = [N//2+xi for xi in range(E-N+1)]
        mov_ave_x = [N//2+xi for xi in range(len(mov_ave))]
        if 'val' in m:
            print('plot is in red')
            if 'loss' in m:
                # validation loss
                vloss = [1.0*p for p in parms_dict[m]]
                plt.plot(vloss,'r',linewidth=0.5)
                plt.plot(mov_ave_x,1.0*mov_ave,'r',linewidth=2)
            else:
                # validation accuracy
                plt.plot(parms_dict[m],'r',linewidth=0.5)
                plt.plot(mov_ave_x,mov_ave,'r',linewidth=2, label='validation metric')
        else:
            print('plot is in green')
            if 'loss' in m:
                # training loss
                plt.plot(parms_dict[m],'g',linewidth=0.5)
                plt.plot(mov_ave_x,mov_ave,'g',linewidth=2)
            else:
                # training accuracy
                plt.plot(parms_dict[m],'g',linewidth=0.5)
                plt.plot(mov_ave_x,mov_ave,'g',linewidth=2, label='training metric')
        if plt_key[mi]==1:
            plt.title('$\it{}$'.format(title),fontsize=15)
            pass
        if plt_key[mi]==2:
            # plt.title('Validation',fontsize=10)
            pass
        if plt_key[mi]>1:
            # legend = plt.legend(legend_label,fontsize=12,loc='lower right')
            legend = plt.legend(fontsize=12,loc='lower right')
            pass
        if mi < 2:
            plt.ylabel(m,fontsize=15)
            # plt.ylabel(m.replace('acc','accuracy').replace('mean_squared_error','mse'),fontsize=15)
        if plt_key[mi]==2:
            plt.xlabel('training epochs',fontsize=15)
        else:
            ax.set_xticklabels([])
        plt.grid(True)
        plt.tight_layout()
        ymax = 1
        if 'loss' in m:
            # constant for comparison experiments
            ymax = 2.0 # 1.1*max_loss
        # plt.ylim(0,ymax)
        plt.xlim(0,E)
        if 'acc' in m:
            # plt.text(0.05*E,0.2*1.0,lb2,fontsize=15)
            plt.text(0.05*E,0.991,lb2,fontsize=15)
        else:
            # plt.text(0.05*E,ymax*0.8,lb1,fontsize=15)
            plt.text(0.05*E,0.005,lb1,fontsize=15)
    fn = os.path.join(save_dir,'{}_performance_metrics_2by1_zoomed.png'.format(experiment))
    # fn = os.path.join(save_dir,'{}_performance_metrics_2by1.png'.format(experiment))
    print('Saving performance plot to : {}'.format(fn))
    plt.savefig(fn)

    legend.remove()
    fn = os.path.join(save_dir,'{}_performance_metrics_nolegend.png'.format(experiment))
    print('Saving performance plot to : {}'.format(fn))
    plt.savefig(fn)
    plt.close(fig_nb)


def get_parms_dict(experiment):
    # hist_dir = '/data/rpizarro/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
    hist_dir = '/trials/data/rpizarro/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
    hist_files = glob.glob(os.path.join(hist_dir,'history*parms.json'))

    if len(hist_files)>1:
        hist_files.sort()

        # initializing dictionary keys
        parms_dict = get_parms(hist_files[0])
        print(parms_dict)
        # plt_key = {0:2,1:4,2:6,3:1,4:3,5:5}

        for fn in hist_files[1:]:
            curr_dict = get_parms(fn)

            for metric in parms_dict.keys():
                parms_dict[metric] += curr_dict[metric]
    else:
        print(hist_files)
        parms_dict = get_parms(hist_files[0])
        print(parms_dict.keys())

    if 'mse' in parms_dict.keys():
        parms_dict.pop('mse')
        parms_dict.pop('val_mse')
    else:
        parms_dict.pop('mean_squared_error')
        parms_dict.pop('val_mean_squared_error')
    return parms_dict

def check_experiment_repeats(experiment):
    hist_dir_star = '/data/rpizarro/datasets/shared/rpizarro/noise/weights/{}*'.format(experiment)
    experiments = sorted(glob.glob(hist_dir_star))
    if len(experiments) > 1:
        print(experiments)

    return experiments


# python noise.plot_performance.py rap_NN008_multiple_artifact clean_percent_098 XV0 nb_classes_02 nb_samples_factor_01.00 001-initialized_constant_clean_098_ep0050 'title' '(a)' '(b)'
exp_dirs = sys.argv[1:-3]
title = sys.argv[-3]
lb1 = sys.argv[-2]
lb2 = sys.argv[-1]
experiment = '/'.join(exp_dirs)

# experiments = check_experiment_repeats(experiment)
# sys.exit()
parms_dict = get_parms_dict(experiment)

# ['acc', 'loss', 'val_acc', 'val_loss']
plt_key = {0:2,1:1,2:2,3:1}

experiment_fn = experiment.replace('/','_')

if '002-ramp_clean_050_to_098_ep0500' in experiment:
    plot_cp_parms(experiment_fn,parms_dict,plt_key=plt_key,title=title,lb1=lb1,lb2=lb2)
else:
    plot_parms(experiment_fn,parms_dict,plt_key=plt_key,title=title,lb1=lb1,lb2=lb2)


