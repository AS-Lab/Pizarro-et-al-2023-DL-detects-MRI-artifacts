import pandas as pd
import os,sys



save_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended/'
fn = os.path.join(save_dir,'noise_scans.tj.csv')

columns = ['intensity', 'motion', 'coverage']

noise = pd.read_csv(fn,usecols=columns)

print(noise.head())
print(list(noise))

noise['max_val'] = noise.max(axis=1)
noise = noise[columns].div(noise.max_val, axis=0).round(1)

print(noise.head())

noise = noise.sort_values(by =columns,ascending =[False,False,False])
print(noise)

noise_f = noise
noise_f[noise_f<0.8]=0
print(noise_f)
print(0.2*noise_f.sum())
print(0.6*noise_f.sum())
# sys.exit()

noise_grouped = noise.groupby(columns).size().reset_index().rename(columns={0:'count'})
noise_grouped = noise_grouped.sort_values(by=['count'],ascending=[False]).reset_index(drop=True)
noise_grouped['cumulative sum'] = noise_grouped['count'].cumsum()
noise_grouped['% used'] = 100.0*noise_grouped['cumulative sum']/noise_grouped['count'].sum()
noise_grouped['% used'] = noise_grouped['% used'].round(2)
print(noise_grouped)

fn = os.path.join(save_dir,'noise_scans.grouped.tj.csv')
print('Saving tabulated grouped count for noise scans to : {}'.format(fn))
noise_grouped.to_csv(fn)


