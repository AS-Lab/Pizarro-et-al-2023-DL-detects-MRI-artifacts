import pandas as pd
import os,sys
import glob

def get_modalities(trial):

    modalities_fn = os.path.join('/data/datasets',trial,'modalities.csv')
    mod_df = pd.read_csv(modalities_fn)
    return list(mod_df)
    

def get_fn_match(path,mod):

    fn_str = path.replace('/data/datasets/','').replace('/','_')
    return fn_str

def gen_used_trials(df):
    paths = df['path']
    trials = list(set([p.split('/')[3] for p in paths]))
    trials.sort()
    return trials


def gen_noise_trials():
    save_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended/'
    fn = os.path.join(save_dir,'noise_scans.csv')
    noise = pd.read_csv(fn)
    noise_trials = gen_used_trials(noise)
    print('The scans with artifacts are from {} trials : {}'.format(len(noise_trials),noise_trials))
    return noise_trials



# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# Before running this script you have to run: noise.save.noise_set.py
noise_trials = gen_noise_trials()

clean_dirs_fn = '/home/rpizarro/noise/XValidFns/trials_combined_clean_paths.csv'
df_clean_dirs = pd.read_csv(clean_dirs_fn)
print(df_clean_dirs.head())

cols_mnc = ['trial','path_dir','modality','mnc_file']
df_mnc = pd.DataFrame(columns=cols_mnc)

for index, row in df_clean_dirs.iterrows():
    if row.trial not in noise_trials:
        print('Trial {} not in noise trials'.format(row.trial))
        # sys.exit()
        continue
    modalities = get_modalities(row.trial)
    for m in modalities:
        mnc_fn_match = get_fn_match(row.path,m)
        mnc_files = glob.glob(os.path.join(row.path,'{}_{}_original.mnc.gz'.format(mnc_fn_match,m)))
        if not mnc_files:
            mnc_files = glob.glob(os.path.join(row.path,'{}_{}.mnc.gz'.format(mnc_fn_match,m)))
        if len(mnc_files)<1:
            print('We did not find a unique number of mnc files {} : {} : {}'.format(row.path,m,mnc_files))
            continue
        elif len(mnc_files)>1:
            print('We found too many mnc files {} : {} : {}'.format(row.path,m,mnc_files))
            continue
        else:
            print(mnc_files[0])
            append_list = [row.trial,row.path,m,mnc_files[0]]
            row_append = pd.DataFrame([append_list], columns=list(df_mnc))
            df_mnc = df_mnc.append(row_append, ignore_index=True)

df_fn = os.path.join(os.path.dirname(clean_dirs_fn),'mnc_clean_paths.csv')
print('Writing combined sheet to : {}'.format(df_fn))
df_mnc.to_csv(df_fn)



