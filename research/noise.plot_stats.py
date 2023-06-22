import pandas as pd
import glob
import os,sys
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

epoch_transience_dir = '/home/rpizarro/noise/prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/002-ramp_clean_050_to_098_ep0500/epoch_transience'


ep_dirs = glob.glob(os.path.join(epoch_transience_dir,'ep0*'))

cols = ['epoch','AUC','accuracy','sensitivity','specificity']
df_stats = pd.DataFrame(columns=cols)

for ed in sorted(ep_dirs):
    ep = int(os.path.basename(ed).replace('ep',''))
    # print(ep)
    df = pd.read_csv(os.path.join(ed,'summary_stats.threshold_by_unc.csv'))
    row_vals = [ep,df['AUC'][0],df['accuracy'][0],df['sensitivity'][0],df['specificity'][0]]
    row = pd.DataFrame([row_vals],columns=cols)
    df_stats = df_stats.append(row,ignore_index=True)
    
fig_nb=1001
plt.figure(fig_nb,figsize=(6,12))
plt.subplot(411)
plt.plot(df_stats['epoch'],df_stats['AUC'],'*-')
plt.tick_params(labelsize=15)
plt.ylim([0.95,1.0])
plt.ylabel('AUC',fontsize=20)
plt.tight_layout()
plt.subplot(412)
plt.plot(df_stats['epoch'],df_stats['accuracy'],'*-')
plt.tick_params(labelsize=15)
plt.ylim([0.8,1.0])
plt.ylabel('accuracy',fontsize=20)
plt.tight_layout()
plt.subplot(413)
plt.plot(df_stats['epoch'],df_stats['sensitivity'],'*-')
# plt.plot(df_stats['epoch'],df_stats['specificity'])
plt.tick_params(labelsize=15)
plt.ylim([0.8,1.0])
plt.ylabel('specificity',fontsize=20)
plt.tight_layout()
plt.subplot(414)
plt.plot(df_stats['epoch'],df_stats['specificity'],'*-')
# plt.plot(df_stats['epoch'],df_stats['sensitivity'])
plt.tick_params(labelsize=15)
plt.ylim([0.8,1.0])
plt.ylabel('sensitivity',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.tight_layout()

save_dir = '/home/rpizarro/noise/figs/002-ramp_clean_050_to_098_ep050/epoch_transience'
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)
fn = os.path.join(save_dir,'stats_by_ramp.png')
print('Saving to : {}'.format(fn))
plt.savefig(fn)
plt.close(fig_nb)



