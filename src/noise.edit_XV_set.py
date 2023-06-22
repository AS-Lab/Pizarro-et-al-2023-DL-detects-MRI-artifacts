import pandas as pd
import numpy as np
import os,sys
from collections import Counter
import random
import glob


def gen_used_trials(df):
    paths = df['path']
    # paths = [p.replace('/FAILEDSCANS/trials','') for p in paths]
    trials = list([p.replace('/FAILEDSCANS/trials','').split('/')[3] for p in paths])
    d = Counter(trials)
    trials_set,trials_count = d.keys(), d.values()
    trials.sort()
    return list(trials_set),list(trials_count)


def gen_trials_used_df(noise,clean):
    noise_trials_set,noise_trials_count = gen_used_trials(noise)
    dict_noise = {'trial':noise_trials_set,'noise':noise_trials_count}
    noise_trials_df = pd.DataFrame(dict_noise)

    clean_trials_set,clean_trials_count = gen_used_trials(clean)
    dict_clean = {'trial':clean_trials_set,'clean':clean_trials_count}
    clean_trials_df = pd.DataFrame(dict_clean)

    trials_used = noise_trials_df
    trials_used[['clean']] = clean_trials_df[['clean']]
    trials_used['clean'] = trials_used['clean'].fillna(0)

    trials_used['clean'] = trials_used['clean'].astype(int)
    return trials_used



def append_still_missing(trials_used,N=2):
    # may work as is for small N < 5 or so
    noise_col = trials_used['noise']
    trials_used.loc[trials_used['clean']==0,'noise']=0
    trials_used['prob'] = N*trials_used['noise']/trials_used['noise'].sum()
    trials_used['still_missing'] = trials_used['prob'].nlargest(N)
    trials_used['still_missing'] = trials_used['still_missing'].fillna(0).apply(np.ceil).astype(int)
    trials_used = trials_used.drop(columns=['prob'])
    trials_used['noise'] = noise_col
    return trials_used


def append_paths(df,clean_path_append):
    print('We will append the following paths')
    print(clean_path_append)

    print(df)


    nb = len(clean_path_append)
    dict_append = {'clean':nb*[1.0],'artifact':nb*[0.0],'path':clean_path_append}
    append_df = pd.DataFrame(dict_append)
    print(append_df)

    df = df.append(append_df,ignore_index=True)

    print(df)

    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/valid.art123.csv'
    print('We already saved valid spreadsheet to : {}'.format(fn))
    # df.to_csv(fn)


def get_cp_2_append(trials_used,df_source):
    # this was developed for the validation set only to append a couple of trials
    trials_missing = list(trials_used.loc[trials_used['still_missing']==1]['trial'])
    # print(trials_missing)

    clean_source = df_source.iloc[139:].reset_index(drop=True)

    clean_path_append = []

    for t in trials_missing:
        clean_paths = [p for p in clean_source['path'] if t in p]
        random.shuffle(clean_paths)
        missing = 0
        for cp in clean_paths:
            # print(cp)
            if cp not in set(df['path']):
                clean_path_append += [cp]
                missing += 1
                still_missing = list(trials_used.loc[trials_used['trial']==t]['still_missing'])
                if missing < still_missing[0]:
                    continue
                else:
                    break
        # print(len(clean_trial))
    return clean_path_append




def update_valid():
    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/valid.art123.ep0000.csv'
    df = pd.read_csv(fn,index_col=0)

    save_dir = '/home/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used'
    noise = df.iloc[:139]
    clean = df.iloc[139:].reset_index(drop=True)


    fn_source = '~/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/valid.art123.csv'
    df_source = pd.read_csv(fn_source,index_col=0)


    # this is the same as noise
    noise_source = df_source.iloc[:139]
    noise_source = noise_source.sort_values(by=['artifact', 'path']).reset_index(drop=True)


    trials_used = gen_trials_used_df(noise,clean)
    # currently set up for a few missing images like N < 5
    trials_used = append_still_missing(trials_used,N=2)
    print(trials_used)


    # this is for the validation set
    clean_path_append = get_cp_2_append(trials_used,df_source)
    append_paths(df,clean_path_append)

    # clean_trials = clean_source
    # print(clean_source)


def get_nb_samples(df,nb_artifacts=3):
    # returns a list nb_samples by summing categorical values
    nb_samples = df.iloc[:,:nb_artifacts+1].sum().tolist()
    return nb_samples


def get_trials_from_path(df):
    freq = {}
    for f in df['path']:
        trial = f.split('/')[3]
        if trial in freq:
            freq[trial] += 1
        else:
            freq[trial] = 1
    trials_df = pd.DataFrame.from_dict(freq,orient='index',columns=['nb_artifact'])
    return trials_df

def select_by_trails(clean,nb_clean,trials_df):
    factor_mult = nb_clean/trials_df['nb_artifact'].sum()
    trials_df['nb_clean'] = factor_mult*trials_df['nb_artifact']
    trials_df['nb_clean'] = trials_df['nb_clean'].round(0).astype(int)
    clean_df = pd.DataFrame(columns=list(clean))
    freq = {}
    for index,row in clean.iterrows():
        trial = row['path'].split('/')[3]
        if trial not in trials_df.index:
            continue
        if trial not in freq:
            # we have a new trial entry
            clean_df = clean_df.append(row,ignore_index=True)
            freq[trial] = 1
        elif freq[trial] < trials_df['nb_clean'][trial]:
            clean_df = clean_df.append(row,ignore_index=True)
            freq[trial] += 1
        else:
            # Too many from this trial
            continue

    return clean_df



def update_test(clean_ratio=0.5):
    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/test.art123.csv'
    df = pd.read_csv(fn,index_col=0)
    print('\nThe full testing dataset including artifacts and path is size : {}'.format(df.shape))
    df = df.sort_values(by=list(df)).reset_index(drop=True)
    print(df)

    nb_classes = int(df.shape[1]-1)
    nb_samples = get_nb_samples(df,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    nb_clean = int(clean_ratio*nb_artifact/(1 - clean_ratio))
    print('We will manually change to corresponding clean_ratio of {0:0.3f}, by using size [clean,artifact] : [{1},{2}]'.format(clean_ratio,nb_clean,nb_artifact))

    artifact = df.iloc[:nb_artifact]
    trials_df = get_trials_from_path(artifact)

    clean = df.iloc[nb_artifact:]
    # sort clean to get same scans each time
    clean = clean.sort_values('path').reset_index(drop=True)
    # DO NOT SHUFFLE # clean = clean.sample(frac=1).reset_index(drop=True)

    # manual hack for new artifact data
    clean = select_by_trails(clean,int(nb_clean/0.79),trials_df)
    clean = clean.sample(frac=1).reset_index(drop=True)
    clean = clean.iloc[:139]
    clean = clean.sort_values(by=list(clean)).reset_index(drop=True)

    print(clean)
    clean_ratio_actual = float(clean.shape[0]) / (artifact.shape[0] + clean.shape[0])
    print('We went through the list of files available and found size [clean,artifact] : [{0},{1}] resulting in actual clean_ratio : {2:0.3f}'.format(clean.shape[0],artifact.shape[0],clean_ratio_actual))
    test = artifact.append(clean,ignore_index=True)

    print(test)
    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/test.art123.csv'
    print('We already saved test spreadsheet to : {}'.format(fn))
    # test.to_csv(fn)




def update_98_valid():
    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/valid.art123.csv'
    df = pd.read_csv(fn,index_col=0)
    print(df)
    nb_classes = int(df.shape[1]-1)
    nb_samples = get_nb_samples(df,nb_classes-1)
    print(nb_samples)
    nb_artifact = int(nb_samples[-1])
    artifact = df.iloc[:nb_artifact]
    clean = df.iloc[nb_artifact:]

    fn_50 = '~/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/valid.art123.csv'
    df_50 = pd.read_csv(fn_50,index_col=0)
    # print(df_50)
    nb_samples = get_nb_samples(df_50,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    clean_50 = df_50.iloc[nb_artifact:]
    
    remove_count = 0
    paths_remove = []
    clean_paths = list(clean['path'])
    random.shuffle(clean_paths)
    for cp in clean_paths:
        if cp not in set(clean_50['path']):
            paths_remove += [cp]
            remove_count += 1
            if remove_count == 20:
                break
    
    clean = clean[~clean['path'].isin(paths_remove)]
    df = artifact.append(clean).reset_index(drop=True)
    print(df)
    print(df.shape)
    print('We removed 20 clean images that are not in : {}'.format(fn_50))
    print('We already overwrote to : {}'.format(fn))
    # df.to_csv(fn)
    
    # print(paths_remove)

    
def update_98_test():
    fn = '~/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/test.art123.csv'
    df = pd.read_csv(fn,index_col=0)
    nb_classes = int(df.shape[1]-1)
    nb_samples = get_nb_samples(df,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    artifact = df.iloc[:nb_artifact]
    clean = df.iloc[nb_artifact:]

    fn_50 = '~/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/test.art123.csv'
    df_50 = pd.read_csv(fn_50,index_col=0)
    nb_samples = get_nb_samples(df_50,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    clean_50 = df_50.iloc[nb_artifact:]
    
    remove_count = 0
    paths_remove = []
    clean_paths = list(clean['path'])
    random.shuffle(clean_paths)
    for cp in clean_paths:
        if cp not in set(clean_50['path']):
            paths_remove += [cp]
            remove_count += 1
            if remove_count == 20:
                break
    
    clean = clean[~clean['path'].isin(paths_remove)]
    df = artifact.append(clean).reset_index(drop=True)
    print(df)
    print(df.shape)
    print('We removed 20 clean images that are not in : {}'.format(fn_50))
    print('We already overwrote to : {}'.format(fn))
    # df.to_csv(fn)
    
    # print(paths_remove)


def update_predictions(sub_experiment):

    fn = '/home/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/valid.art123.csv'
    df_valid = pd.read_csv(fn,index_col=0)
    
    print('Working on sub_experiment : {}'.format(sub_experiment))
    pred_dir = '/home/rpizarro/noise/prediction/rap_NN008_multiple_artifact/clean_percent_098/XV0/nb_classes_02/nb_samples_factor_01.00/'

    valid_dir = os.path.join(pred_dir,sub_experiment,'valid')
    print('With path : {}'.format(valid_dir))
    
    mc_files = glob.glob(os.path.join(valid_dir,'mc*.csv'))
    for idx,mc in enumerate(sorted(mc_files)):
        print(mc)
        df = pd.read_csv(mc,index_col=0)
        if not idx:
            print(df)
            valid_files = list(df_valid['path'])
            df_remove = df[~df['path valid'].isin(valid_files)]
            remove_indices = df_remove.index
            if not list(remove_indices):
                print('We already removed the pertinent files... going to next one')
                break
        df = df[~df.index.isin(remove_indices)].reset_index(drop=True)
        # print(df.shape)
        print('Will overwrite mc prediction csv spreadsheet')
        df.to_csv(mc)
        # print(mc)


def update_predictions_50(sub_experiment):
    # we should add mc predictions for two missing scans and incorporate their predictions for the mc files

    fn = '/home/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_050/XV0_used/valid.art123.csv'
    df_valid = pd.read_csv(fn,index_col=0)
    
    print('Working on sub_experiment : {}'.format(sub_experiment))
    pred_dir = '/home/rpizarro/noise/prediction/rap_NN008_multiple_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_01.00/'

    valid_dir = os.path.join(pred_dir,sub_experiment,'valid')
    print('With path : {}'.format(valid_dir))
    
    mc_files = glob.glob(os.path.join(valid_dir,'mc*.csv'))




def update_multiple_predictions():
    sub_experiment_list = [
            '002-E1ramp_020_data_800_to_20000_ep0060',
            '002-E1ramp_005_data_800_to_20000_ep0053',
            '001-random_constant_clean_098_ep0050',
            '001-initialized_constant_clean_098_ep0050',
            '002-E1ramp_005_data_800_to_20000_ep0053_N2',
            '002-E1ramp_100_data_800_to_20000_ep0100_N2',
            '001-initialized_constant_clean_098_ep0050_N2'
            ]

    for sub_experiment in sub_experiment_list:
        update_predictions(sub_experiment)



# sub_experiment = sys.argv[1]    

# update_valid()
# update_test(clean_ratio=0.5)


# update_98_valid()
# update_98_test()

# update_multiple_predictions()
# Next we should do the same for 50 percent clean and then redo secondary analysis





