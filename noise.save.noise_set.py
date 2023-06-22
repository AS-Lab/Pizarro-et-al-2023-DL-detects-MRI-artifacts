import pandas as pd
import numpy as np
import os,sys
import nibabel as nib
import random
import time
from itertools import groupby
from itertools import islice
from collections import Counter
# import scipy.ndimage
np.seterr(all='raise')
pd.set_option('display.width', 1000)

def getPathsLabel(noise_csv,artifacts, level):
    noise_paths=[]
    noise_labels=[]
    # weighted artifact level
    level_weight = {1:0.3,2:0.6,3:1.0}
    df = pd.read_csv(noise_csv,converters={'mod':int,'artifact':int,'level':int})
    for index,row in df.loc[(df['mod']!=8) & (df['artifact'].isin(artifacts)) & (df['level'].isin(level)) & (~df['path'].str.contains('NOT_AVAILABLE'))].iterrows():
        if can_be_opened(row['path']):
            print(str(row.values.tolist()))
            noise_paths.append(row['path'])
            label = [0]*(1+len(artifacts))
            label[row['artifact']] = level_weight[row['level']]
            print(label)
            noise_labels.append(label)
            print('noise list length : {}'.format(len(noise_labels)))
            if len(noise_labels) > 200000:
                break
                # return noise_paths, noise_labels
        else:
            continue
    print("Noise entries: %d labels: %d" %(len(noise_paths),len(noise_labels)))
    c = list(zip(noise_paths,noise_labels))
    random.shuffle(c)
    noise_paths, noise_labels = zip(*c)
    return noise_paths, noise_labels
            

def can_be_opened(full_fname):
    ret_val=True
    try:
        nib.load(full_fname)
    except:
        ret_val=False
        print("Failed to open the file {}".format(full_fname))
    return ret_val


def set_to_df(paths,labels,cols=['clean','intensity','motion','coverage','path']):
    df = pd.DataFrame(columns=cols)
    for p,l in zip(paths,labels):
        print(p,l)
        if df['path'].isin([p]).any():
            print('path entry exists updating row...')
            idx = l.index(max(l))
            df.loc[df.path == p, cols[idx]] = df.loc[df.path == p, cols[idx]] + l[idx]
        else:
            row = pd.DataFrame([l+[p]], columns=cols)
            df = df.append(row, ignore_index=True)
    return df


def exclude_moderates(df):
    # we only keep the samples that have at least one column equal to 1.0
    level = 1.0
    return df.loc[(df.clean == level) | (df.intensity == level) | (df.motion == level) | (df.coverage == level)]

def normalize_noise(df):
    col_list = list(df)
    col_list.remove('path')
    df['sum'] = df[col_list].sum(axis=1)
    for c in col_list:
        df[c] = df[c]/df['sum'].round(3)
    return df.drop(columns=['sum'])


def gen_used_trials(df):
    paths = df['path']
    trials = list([p.split('/')[3] for p in paths])
    d = Counter(trials)
    trials_set,trials_count = d.keys(), d.values()
    trials.sort()
    print(trials)
    # trials_set = set(trials)
    # trials_count = [len(list(group)) for key, group in groupby(trials)]

    # trials = list(set([p.split('/')[3] for p in paths]))
    # trials.sort()
    return list(trials_set),list(trials_count)


start_time = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# change the model file in get_model function

view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# fail_paths_fn = '/home/rpizarro/noise/sheets/failed.hazpaths.log'
# fail_labels_fn = '/home/rpizarro/noise/sheets/failed.noiselabels.csv'
# fail_paths_fn = '/home/rpizarro/noise/sheets/dataset_list29-06-12-19-06.txt'
fail_paths_fn = '/home/rpizarro/noise/sheets/dataset_list07-08-07-10-20.csv'

# print("Empty Directories:")
# List of artifacts: Geometric,Intensity,Movement,Coverage,Contrast,Acquisition,MissScan,Other
# Let's try just Intensity (78) and Movement (70) for now...
# artifacts_to_model = [1,2,3] # [0,2,3,4,6] # range(5) + [6]

# artifacts: 0-geometric distortion, 1-intensity, 2-motion/ringing, 3-coverage, 4-7 other stuff
artifacts_to_model = [1,2,3] # 0-7
# level 1-not visible, 2-subte, 3-prominent
artefact_level = [1,2,3]

# [noise_paths,noise_labels]=getPathsLabel(fail_paths_fn,fail_labels_fn,artifacts_to_model)
[noise_paths,noise_labels]=getPathsLabel(fail_paths_fn,artifacts_to_model,artefact_level)
# print("")

cols=['clean','intensity','motion','coverage','path']
noise = set_to_df(noise_paths,noise_labels,cols=cols)
print(noise)
print(noise.shape)

noise = exclude_moderates(noise)
print(noise)
print(noise.shape)

noise = normalize_noise(noise)
print(noise)
print(noise.shape)

save_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended/'
fn = os.path.join(save_dir,'noise_scans.csv')
print('Saving noise scans to : {}'.format(fn))
noise.to_csv(fn,index_label='index')


noise_trials_set,noise_trials_count = gen_used_trials(noise)
print('The scans with artifacts are from {} trials : {}'.format(len(noise_trials_set),noise_trials_set))
print('with frequency: {}'.format(noise_trials_count))
dict_noise = {'trial':noise_trials_set,'noise':noise_trials_count}
noise_trials_df = pd.DataFrame(dict_noise)
fn = os.path.join(save_dir,'trials_used_noise.csv')
print('Saving trials used count to : {}'.format(fn))
noise_trials_df.to_csv(fn,index_label='index')


print("--- Time :: Noise paths : %s seconds ---" % (time.time() - start_time))


