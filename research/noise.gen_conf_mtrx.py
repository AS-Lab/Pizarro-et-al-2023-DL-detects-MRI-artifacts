import numpy as np
import matplotlib.pyplot as plt
import itertools



def plot_conf(conf,fn,outcomes=['clean','artifact'],index_mode=True,title=None):
    plt.figure(figsize=(20,20))
    cmap=plt.cm.Blues
    fnt_size = 80
    if conf.shape[0]<3:
        fnt_size = 100
    # fig=plt.figure(fig_nb,figsize=(5,5))
    imgplot=plt.imshow(conf, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(outcomes))
    # cbar = plt.colorbar(imgplot,fraction=0.046, pad=0.04) # ticks=[0,10,20,30,40,50])
    # cbar.ax.tick_params(labelsize=fnt_size)
    plt.xticks(tick_marks, outcomes, rotation=45, fontsize=fnt_size)
    plt.yticks(tick_marks, outcomes, fontsize=fnt_size)
    plt.ylabel('Inferred artifact',fontsize=1.2*fnt_size)
    plt.xlabel('Target artifact',fontsize=1.2*fnt_size)
    if title:
        plt.title(title,fontsize=1.2*fnt_size)
    plt.tight_layout()
    # print(conf)
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
    print('Saving to : {}'.format(fn))
    plt.savefig(fn)
    plt.close()



conf = np.array([[6102,28],[717,111]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/initial_attempt_realistic/confusion.png'

# plot_conf(conf,fn,index_mode=True,title='')

conf = np.array([[20,0],[0,107]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN007_single_artifact/XV0/S1_confusion.png'

# plot_conf(conf,fn,index_mode=True,title='')

conf = np.array([[4056,69],[2762,71]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/test/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal/prob_threshold_87.png'

# plot_conf(conf,fn,index_mode=True,title='')

conf = np.array([[6818,140],[0,0]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/test/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal/index_amax.png'

# plot_conf(conf,fn,index_mode=True,title='')


conf = np.array([[19,14],[1,93]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_stg1/XV0/test/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_stg1_ep0200_NN7_combat/prob_threshold_01.png'

# plot_conf(conf,fn,index_mode=True,title='')

conf = np.array([[5165,5],[25,62]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/valid/nb_classes_02/nb_samples_factor_01.00/002-E1ramp_100_data_800_to_20000_ep0100/prob_97_s2_01.png'

# plot_conf(conf,fn,index_mode=True,title='')




# conf = np.array([[tn,fn],[fp,tp]])
conf = np.array([[6355,32],[463,108]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/test/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal/confusion_new_GPU_by5.png'

plot_conf(conf,fn,index_mode=True,title='')

conf = np.array([[6098,22],[720,118]])
fn = '/trials/data/rpizarro/noise/figs/performance/mc/rap_NN008_multiple_artifact/clean_percent_098/XV0/test/nb_classes_02/nb_samples_factor_01.00/001-random_constant_clean_098_ep0050_focal/confusion_new_GPU_by10.png'

plot_conf(conf,fn,index_mode=True,title='')





