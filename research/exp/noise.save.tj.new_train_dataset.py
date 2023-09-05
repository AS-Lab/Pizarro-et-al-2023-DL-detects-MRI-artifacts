import pandas as pd
import os


def load_df(XV_dir):

    train_fn = os.path.join(XV_dir,'train.art123.csv')
    train_df = pd.read_csv(train_fn,index_col=0)
    valid_fn = os.path.join(XV_dir,'valid.art123.csv')
    valid_df = pd.read_csv(valid_fn,index_col=0)
    test_fn = os.path.join(XV_dir,'test.art123.csv')
    test_df = pd.read_csv(test_fn,index_col=0)

    df = train_df.append([valid_df, test_df],ignore_index=True)

    return df

def manual_split(df):
    # train ratio to split
    tr=5
    df0 = df.iloc[lambda x: x.index % tr == 0].reset_index(drop=True)
    df1 = df.iloc[lambda x: x.index % tr == 1].reset_index(drop=True)
    df2 = df.iloc[lambda x: x.index % tr == 2].reset_index(drop=True)
    df3 = df.iloc[lambda x: x.index % tr == 3].reset_index(drop=True)
    df4 = df.iloc[lambda x: x.index % tr == 4].reset_index(drop=True)

    return df0,df1,df2,df3,df4

def group_df(df0,df1,df2,df3,df4):
    train_df = df0.append([df1, df2],ignore_index=True)
    return train_df,df3,df4


tj_path = '/home/rpizarro/noise/sheets/tj'
fn = os.path.join(tj_path,'combined_sets.tj_filter.csv')
print('Loading file : {}'.format(fn))
df_keep = pd.read_csv(fn,index_col=0)

XV_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended'
df_pre_tj = load_df(XV_dir)

# remove files not in df_keep and reshuffle
df = df_pre_tj[df_pre_tj.path.isin(df_keep.path)]
df = df.sample(frac=1).reset_index(drop=True)

# split into five parts
df0,df1,df2,df3,df4 = manual_split(df)

for XV in range(5):
    save_dir = os.path.join(XV_dir,'tj','XV{}'.format(XV))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    # for each XV we will rotate the test and valid dataset
    train_df,valid_df,test_df = group_df(df0,df1,df2,df3,df4)

    train_fn = os.path.join(save_dir,'train.art123.csv')
    print('Saving train dataset to : {}'.format(train_fn))
    train_df.to_csv(train_fn)
    valid_fn = os.path.join(save_dir,'valid.art123.csv')
    print('Saving valid dataset to : {}'.format(valid_fn))
    valid_df.to_csv(valid_fn)
    test_fn = os.path.join(save_dir,'test.art123.csv')
    print('Saving test dataset to : {}'.format(test_fn))
    test_df.to_csv(test_fn)
    # rotate parts for following XV set
    df0,df1,df2,df3,df4 = df1,df2,df3,df4,df0


