import pandas as pd
import os

# we will start working in a new dir: single_artifact
save_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended/'
fn = os.path.join(save_dir,'noise_scans.csv')

columns = ['clean','intensity', 'motion', 'coverage']

noise = pd.read_csv(fn,index_col=0).reset_index(drop=True)

print(noise.head())
print(list(noise))

noise['max_val'] = noise.max(axis=1)
noise = noise[columns].div(noise.max_val, axis=0).round(1).join(noise[['path']])

print(noise.head())



fn = os.path.join(save_dir,'noise_scans.grouped.csv')
noise_grouped = pd.read_csv(fn,index_col=0)

print(noise_grouped)
threshold = 90.0
noise_grouped = noise_grouped[noise_grouped['% used'] < threshold]
noise_categories = noise_grouped[columns[1:]]
print(noise_categories)

noise['top_noise'] = 0
print(noise.head())
for index,row in noise_categories.iterrows():
    noise.loc[(noise.intensity == row['intensity']) 
            & (noise.motion == row['motion']) 
            & (noise.coverage == row['coverage']),'top_noise'] += 1

noise = noise.loc[noise.top_noise > 0].drop(columns=['top_noise']).reset_index(drop=True)
print(noise)

sing_art_dir = '/home/rpizarro/noise/XValidFns/single_artifact/'
fn = os.path.join(sing_art_dir,'noise_scans.csv'.format(threshold))
print('Writing top categories including {0:0.1f}% to : {1}'.format(threshold,fn))
noise.to_csv(fn)




