import pandas as pd
import os,sys
from collections import Counter

def load_df(XV_dir):

    noise_fn = os.path.join(XV_dir,'noise_scans.csv')
    noise_df = pd.read_csv(noise_fn,index_col=0)

    return noise_df



def gen_used_trials(df):
    paths = df['path']
    trials = list([p.split('/')[3] for p in paths])
    d = Counter(trials)
    trials_set,trials_count = d.keys(), d.values()
    trials.sort()
    print(trials)
    return list(trials_set),list(trials_count)



tj_path = '/home/rpizarro/noise/sheets/tj'
fn = os.path.join(tj_path,'combined_sets.tj_filter.csv')
print('Loading file : {}'.format(fn))
df_keep = pd.read_csv(fn,index_col=0)

XV_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended/'
noise_pre_tj = load_df(XV_dir)

# remove files not in df_keep and reshuffle
noise = noise_pre_tj[noise_pre_tj.path.isin(df_keep.path)]
noise = noise.sample(frac=1).reset_index(drop=True)

print(noise)
noise_fn = os.path.join(XV_dir,'noise_scans.tj.csv')
print('Writing tj_filtered noise set to : {}'.format(noise_fn))
noise.to_csv(noise_fn)


noise_trials_set,noise_trials_count = gen_used_trials(noise)
print('The scans with artifacts are from {} trials : {}'.format(len(noise_trials_set),noise_trials_set))
print('with frequency: {}'.format(noise_trials_count))
dict_noise = {'trial':noise_trials_set,'noise':noise_trials_count}
noise_trials_df = pd.DataFrame(dict_noise)
fn = os.path.join(XV_dir,'trials_used_noise.tj.csv')
print('Saving trials used count to : {}'.format(fn))
noise_trials_df.to_csv(fn,index_label='index')





