import pandas as pd
import os,sys
import glob
import numpy as np
from ast import literal_eval


def load_XV_sets(path):
    train_fn = os.path.join(path,'train_st1_prediction_variance_probability.csv')
    valid_fn = os.path.join(path,'valid_st1_prediction_variance_probability.csv')
    test_fn = os.path.join(path,'test_st1_prediction_variance_probability.csv')
    
    train_df = pd.read_csv(train_fn,index_col=0)
    train_df['XV_set'] = 'train'
    valid_df = pd.read_csv(valid_fn,index_col=0)
    valid_df['XV_set'] = 'valid'
    test_df = pd.read_csv(test_fn,index_col=0)
    test_df['XV_set'] = 'test'

    return train_df,valid_df,test_df

def combine_XV_sets(train_df,valid_df,test_df):
    frames = [train_df,valid_df,test_df]    
    df = pd.concat(frames,ignore_index=True)

    return df


def drop_na(df):
    train_df['zslices'].replace(' ', pd.np.nan, inplace=True)
    train_df.dropna(subset=['zslices'], inplace=True)

    df['Manufacturer'].replace('', pd.np.nan, inplace=True)
    df.dropna(subset=['Manufacturer'], inplace=True)


def clean_train(df):
    # remove unwanted columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # change blank zslice to 0 to later remove it
    df['zslices'].replace(' ', '0', inplace=True)
    # convert string to number
    df['zslices'] = df['zslices'].astype(int)
    # remove 'Medical Corporation' superflouous text
    df['Manufacturer'].replace('Hitachi Medical Corporation', 'Hitachi', inplace=True)
    # df['zslices'] = [literal_eval(item) for item in list(df.zslices)]
    return df


def print_counts(df,cols):
    for c in cols:
        print('Column {}'.format(c))
        print(df[c].value_counts().sort_index())


def left_strip(df,cols):
    for c in cols:
        df[c] = df[c].str.lstrip()
        df[c].replace('',' ',inplace=True)
    return df


def get_mask(df):
    # we are masking the unwanted data
    mask = ((df['zslices'] != 60)
            |(df['Modality'].str.match('DTI'))
            |(df['Modality'].str.match('dwi'))
            |(df['Modality'].str.match('t1g'))
            |(df['Manufacturer'].str.contains('Marconi',case=False))
            |(df['Manufacturer'].str.contains('NotPresent',case=False))
            |(df['Manufacturer'].str.contains(' ',case=False))
            |(df['Manufacturer'].str.contains('Picker',case=False)))

    return mask

def filter_unwanted_scans(df):
    mask = get_mask(df)
    return df.loc[~mask].reset_index(drop=True)

def combine_and_write_XVsets(tj_path):
    train_df,valid_df,test_df = load_XV_sets(tj_path)

    test_df = left_strip(test_df,['Manufacturer'])
    train_df = clean_train(train_df)

    print('\n**Entire Dataset used so far**\n')
    df = combine_XV_sets(train_df,valid_df,test_df)
    print_counts(df,['Manufacturer','Modality','zslices','XV_set'])

    print('\n**Dataframe to keep**\n')
    df_keep = filter_unwanted_scans(df)
    print_counts(df_keep,['Manufacturer','Modality','zslices','XV_set'])

    fn = os.path.join(tj_path,'combined_sets.tj_filter.csv')
    print('Writing dataframe to : {}'.format(fn))
    df_keep.to_csv(fn)


def write_clean_set(tj_path):
    clean_fn = os.path.join(tj_path,'mnc_clean_paths_TJ.csv')
    clean_df = pd.read_csv(clean_fn,index_col=0)
    # this was used to clean train_df above in combine_and_write_XVsets()
    clean_df = clean_train(clean_df)

    print('\n**Entire Dataset used so far**\n')
    print_counts(clean_df,['Manufacturer','Modality','zslices'])

    print('\n**Dataframe to keep**\n')
    df_keep = filter_unwanted_scans(clean_df)
    print_counts(df_keep,['Manufacturer','Modality','zslices'])

    fn = os.path.join(tj_path,'mnc_clean_paths.tj_filter.csv')
    print('Writing dataframe to : {}'.format(fn))
    df_keep.to_csv(fn)

tj_path = '/home/rpizarro/noise/sheets/tj'

combine_and_write_XVsets(tj_path)

write_clean_set(tj_path)

