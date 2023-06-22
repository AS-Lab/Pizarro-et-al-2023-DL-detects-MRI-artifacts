import pandas as pd
import numpy as np
import os,sys
import nibabel as nib
import random
import time
from itertools import islice
# import scipy.ndimage
np.seterr(all='raise')
pd.set_option('display.width', 1000)


def update_trials_used(trials_used,clean_fn):
    noise_sz = trials_used['noise'].sum()
    clean_sz = trials_used['clean_need'].sum()
    df = pd.read_csv(clean_fn,index_col=0)
    clean_available = []
    for index, row in trials_used.iterrows():
        clean_row_by_trial = df.loc[df['trial'] == row['trial']]
        clean_paths = list(set(clean_row_by_trial['path']))
        clean_available += [len(clean_paths)]
        clean_trial_sz = row['clean_need']
        if len(clean_paths) < clean_trial_sz:
            print('We did not find enough files under trial {}'.format(row['trial']))
            clean_trial_sz = len(clean_paths)
            print('We only found {} of {}'.format(clean_trial_sz,row['clean_need']))
    trials_used['clean_available']=clean_available
    trials_used = check_missing(trials_used)
    
    return trials_used

def get_list_clean_paths(trials_used,clean_fn,clean_percent):
    noise_sz = trials_used['noise'].sum()
    clean_sz = trials_used['clean_2_take'].sum()
    print('We want the dataset to be {0:0.2f}% clean, so we want the following size (noise,clean) : ({1},{2})'.format(100*clean_percent,noise_sz,clean_sz))
    df = pd.read_csv(clean_fn,index_col=0)
    clean_paths = []
    for index, row in trials_used.iterrows():
        clean_row_by_trial = df.loc[df['trial'] == row['trial']]
        clean_p_trial = list(set(clean_row_by_trial['path']))
        clean_trial_sz = row['clean_2_take']
        random.shuffle(clean_p_trial)
        clean_paths += clean_p_trial[:clean_trial_sz]
    return clean_paths


def check_missing(trials_used):
    trials_used['clean_2_take']=trials_used[['clean_need','clean_available']].min(axis=1)
    still_missing = trials_used['clean_need'].sum() - trials_used['clean_2_take'].sum()

    if not still_missing:
        trials_used['clean_0_missing'] = 0
        return trials_used
    idx=0
    while still_missing>0:
        idx+=1
        if idx>10:
            print('We have tried updating {} number of times... try reducing clean percentage'.format(idx))
            break
        print('We are missing {} number of files, lets update trials_used'.format(still_missing))
        # untouched noise_col to be reset at the end
        noise_col = trials_used['noise']
        trials_used['clean_diff'] = trials_used['clean_available']-trials_used['clean_2_take']
        # temporary placement to not use for distribution
        trials_used.loc[trials_used['clean_diff']==0,'noise']=0
        prob = still_missing*trials_used['noise']/trials_used['noise'].sum()
        # print(prob)
        # print(prob.argmax())
        if still_missing==1:
            prob[prob.idxmax()]=1
            trials_used['clean_{}_missing'.format(still_missing)] = prob.round(0).astype(int)
        else:
            trials_used['clean_{}_missing'.format(still_missing)] = prob.round(0).astype(int)
        trials_used['clean_2_take'] += trials_used['clean_{}_missing'.format(still_missing)]
        # reset the noise_col
        trials_used['noise']=noise_col
        trials_used = trials_used.drop(columns=['clean_diff'])
        if any(trials_used['clean_2_take'] > trials_used['clean_available']):
            # update clean_2_take with the minimum value
            trials_used['clean_2_take']=trials_used[['clean_2_take','clean_available']].min(axis=1)
        print('We updated trials used column')
        print(trials_used)
        still_missing = trials_used['clean_need'].sum() - trials_used['clean_2_take'].sum()

    if still_missing>0:
        print('We are still missing {} number of files'.format(still_missing))
        print('ERROR : We did not find enough images available by trial. Try lowering the clean_percent.')
        print(trials_used)
        print('File trials_used will NOT be updated and we will exit now')
        sys.exit()
    
    return trials_used


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def get_nb_samples(df):
    nb_samples = df.iloc[:,:-1].sum().tolist()
    return nb_samples


def split_train_valid(noise_clean):
    cols = list(noise_clean)
    noise_clean = noise_clean.sort_values(by=cols[:-1]).reset_index(drop=True)
    print('The full dataset has the following nb_samples, {} : {}'.format(cols[:-1],list(get_nb_samples(noise_clean))))
    # cross validation ratio
    cvr=5
    test = noise_clean.iloc[lambda x: x.index % cvr == 0].reset_index(drop=True)
    valid = noise_clean.iloc[lambda x: x.index % cvr == 1].reset_index(drop=True)
    train = noise_clean.iloc[lambda x: x.index % cvr >= 2].reset_index(drop=True)

    print('The test dataset has the following nb_samples, {} : {}'.format(cols[:-1],list(get_nb_samples(test))))
    print('The valid dataset has the following nb_samples, {} : {}'.format(cols[:-1],list(get_nb_samples(valid))))
    print('The train dataset has the following nb_samples, {} : {}'.format(cols[:-1],list(get_nb_samples(train))))
    
    return train,valid,test


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


def group_artifacts(noise):
    cols = list(noise)
    noise = noise.sort_values(by=cols[:-1]).reset_index(drop=True)
    print(noise)
    maxidx = noise[['intensity','motion','coverage']].idxmax(axis=1)
    noise[['intensity','motion','coverage']]=0

    for index,row in maxidx.iteritems():
        noise.loc[index,row]=1.0
    print(noise)
    print('Entire noise dataset')
    print(noise[['intensity','motion','coverage']].sum())
    print('Training dataset should be around:')
    print(0.6*noise[['intensity','motion','coverage']].sum())
    print('Validation and testing should be:')
    print(0.2*noise[['intensity','motion','coverage']].sum())

    noise['artifact'] = noise[['intensity','motion','coverage']].max(axis=1)
    noise = noise[['clean','artifact','path']].reset_index(drop=True)
    return noise


def check_availability(save_dir,t_used,clean_fn,clean_percent):
    factor_mult = int(round(clean_percent/(1-clean_percent)))
    t_used['clean_need'] = factor_mult*t_used['noise']
    print('We want data to be {0:0.1f}% clean; we need {1} times the noise'.format(100*clean_percent,factor_mult))
    print(t_used)
    t_used = update_trials_used(t_used,clean_fn)
    t_used['available_clean_%'] = 1.0*t_used['clean_available'] / (t_used['noise']+t_used['clean_available'])
    t_used['available_clean_%'] = t_used['available_clean_%'].round(4)
    t_used['clean_%_taken'] = 1.0*t_used['clean_2_take']/(t_used['noise']+t_used['clean_2_take'])
    t_used['clean_%_taken'] = t_used['clean_%_taken'].round(4)
    save_trials_used(save_dir,t_used)

    return t_used

def save_trials_used(save_dir,trials_used):
    fn = os.path.join(save_dir,'trials_used_count.csv')
    print('Saving updated number of files by trial to : {}'.format(fn))
    trials_used.to_csv(fn)
    print(trials_used)



def get_clean_df(trials_used,clean_fn,clean_percent,artifacts_to_model,noise):

    clean_paths = get_list_clean_paths(trials_used,clean_fn,clean_percent)
    print('Clean paths found : {}'.format(len(clean_paths)))

    nb_artifacts=1#len(artifacts_to_model) # or can be changed to noise/no_noise by simply changing this to 1
    clean_label=np.zeros((nb_artifacts+1)).tolist()
    clean_label[0]=1
    clean_labels=[clean_label]*len(clean_paths)

    print(noise.head())
    clean = set_to_df(clean_paths,clean_labels,cols=list(noise))
    print(clean)
    print(clean.shape)

    return clean

def check_overlap(noise,clean):
    start_time = time.time()

    S1 = set(noise.path)
    S2 = set(clean.path)
    item = S1.intersection(S2)
    if item:
        print('ERROR : We found the following path that is present in both lists : {}'.format(item))
        sys.exit()
    else:
        print('GREAT! We found no overlapping paths : {}'.format(item))


    print("--- Time :: Checked if any item is present in both lists : %s seconds ---" % (time.time() - start_time))

def make_XVsets(save_dir,noise,clean):
    start_time = time.time()
    print('Shape of noise directories: {}'.format(noise.shape)) # This only contains the files that are openable
    print('Shape of clean directories: {}'.format(clean.shape)) # This only contains the files that are openable

    noise_clean = noise.append(clean, ignore_index=True)
    print(noise_clean.head())
    print(noise_clean.shape)

    print("--- Time :: Combined noise and clean datasets : %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # This will have the directory and the label
    [train,valid,test]=split_train_valid(noise_clean)


    print("# clean: {} # noise: {} artifact_level: {} artefacts_to_model: {}".format(clean.shape[0],noise.shape[0],str(artifact_level),str(artifacts_to_model)))
    print('train : {}, valid : {}, test : {}'.format(train.shape[0],valid.shape[0],test.shape[0]))
    
    XV0_dir = os.path.join(save_dir,'XV0')
    if not os.path.exists(XV0_dir):
        os.makedirs(XV0_dir)
    print(train.head())
    print(train.shape)
    train_fn = os.path.join(XV0_dir,'train.art123.csv')
    print('Saving train dataset to : {}'.format(train_fn))
    train.to_csv(train_fn)

    print(valid.head())
    print(valid.shape)
    valid_fn = os.path.join(XV0_dir,'valid.art123.csv')
    print('Saving valid dataset to : {}'.format(valid_fn))
    valid.to_csv(valid_fn)

    print(test.head())
    print(test.shape)
    test_fn = os.path.join(XV0_dir,'test.art123.csv')
    print('Saving test dataset to : {}'.format(test_fn))
    test.to_csv(test_fn)

    print("--- Time :: Split for XValid : %s seconds ---" % (time.time() - start_time))




start_time = time.time()
view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

data_dir = '/home/rpizarro/noise/XValidFns/multiple_artifact/'

# Before running this script you have to run: noise.save.noise_set.py
fn = os.path.join(data_dir,'noise_scans.tj.new_trials.csv')
noise = pd.read_csv(fn,usecols=['clean','intensity','motion','coverage','path'])
noise = group_artifacts(noise)
print(noise)

print('We use noise scans that have an artifact as now the problem is two categories: clean x artifact')
# artifacts: 0-geometric distortion, 1-intensity, 2-motion/ringing, 3-coverage, 4-7 other stuff
# artifacts to model is all under one category
artifacts_to_model = [1]#,2,3] # 0-7
# level 1-not visible, 2-subte, 3-prominent
artifact_level = [3] # [1,2,3]

# print('We need to update this file before we can run this script!!!')
fn = os.path.join(data_dir,'trials_used_noise.tj.new_trials.csv')
trials_used = pd.read_csv(fn,index_col=0)
print(trials_used)
print(trials_used['noise'].sum())
# RAP generated usable images based on whether a metric was generated
# ~/noise/src : noise.find_clean_mnc.py, noise.gen_clean_dataset.py
# clean_fn = '/home/rpizarro/noise/XValidFns/mnc_clean_paths.tj.csv'
clean_fn = '/home/rpizarro/noise/sheets/tj/mnc_clean_paths.tj_filter.csv'
clean_percent = 0.50
save_dir = os.path.join(data_dir,'clean_percent_{0:03d}'.format(int(100*clean_percent)))
trials_used = check_availability(save_dir,trials_used,clean_fn,clean_percent)

clean = get_clean_df(trials_used,clean_fn,clean_percent,artifacts_to_model,noise)

print("--- Time :: Clean paths : %s seconds ---" % (time.time() - start_time))

check_overlap(noise,clean)

make_XVsets(save_dir,noise,clean)

