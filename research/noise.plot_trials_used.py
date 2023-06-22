import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os,sys
import numpy as np
import pandas as pd
pd.options.display.width = 0

fn = '/home/rpizarro/noise/XValidFns/prob_level_appended/trials_used_count.csv'


df_trials = pd.read_csv(fn,index_col=0)

print(df_trials)

save_dir = '/home/rpizarro/noise/figs'

df = df_trials.sort_values('trial')

print(df.clean_available)
print(df.clean_available.max())
print(df)
print(list(df))

plt.figure(1)
plt.barh(df.trial,df.clean_available,color=['blue'],label='Available')
plt.barh(df.trial,df.clean_need,color=['orange'],alpha=0.8,label='Needed')
plt.barh(df.trial,df.noise,color=['green'],label='Noisy')
plt.xscale('log')
plt.xlim(1,100000)
plt.gca().invert_yaxis()
plt.legend(loc=1)
plt.tight_layout()

fn = os.path.join(save_dir,'trials_used_barplot-noise_need_available.png')
print('Saving barplot to : {}'.format(fn))
plt.savefig(fn)
plt.close

plt.figure(2)
plt.barh(df.trial,df.clean_need,color=['orange'],alpha=0.8,label='Needed')
plt.barh(df.trial,df.noise,color=['green'],label='Noisy')
plt.xscale('log')
plt.xlim(1,100000)
plt.gca().invert_yaxis()
plt.legend(loc=1)
plt.tight_layout()

fn = os.path.join(save_dir,'trials_used_barplot-noise_need.png')
print('Saving barplot to : {}'.format(fn))
plt.savefig(fn)
plt.close

plt.figure(3)
plt.barh(df.trial,df.noise,color=['green'],label='Noisy')
plt.xscale('log')
plt.xlim(1,100000)
plt.gca().invert_yaxis()
plt.legend(loc=1)
plt.tight_layout()

fn = os.path.join(save_dir,'trials_used_barplot-noise.png')
print('Saving barplot to : {}'.format(fn))
plt.savefig(fn)
plt.close

