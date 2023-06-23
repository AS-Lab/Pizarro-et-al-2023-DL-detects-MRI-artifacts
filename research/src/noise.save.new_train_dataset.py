import os,sys
import pandas as pd


def manual_resplit(train_df):
    # train ratio to split
    tr=3
    df0 = train_df.iloc[lambda x: x.index % tr == 0].reset_index(drop=True)
    df1 = train_df.iloc[lambda x: x.index % tr == 1].reset_index(drop=True)
    df2 = train_df.iloc[lambda x: x.index % tr == 2].reset_index(drop=True)

    return df0,df1,df2

def group_df(df0,df1,df2,df3,df4):
    train_df = df0.append([df1, df2],ignore_index=True)
    return train_df,df3,df4



# This is a one time recross validation to redistribute
# Old split was: train (0,1,2), valid (3), test (4)
# updated split: train (1,2,3), valid (4), test (0)

XV_dir = '/home/rpizarro/noise/XValidFns/'
XV0_dir = os.path.join(XV_dir,'single_artifact','XV0')

train_fn = os.path.join(XV0_dir,'train.art123.csv')
train_df = pd.read_csv(train_fn,index_col=0)
df0,df1,df2 = manual_resplit(train_df)

valid_fn = os.path.join(XV0_dir,'valid.art123.csv')
df3 = pd.read_csv(valid_fn,index_col=0)
test_fn = os.path.join(XV0_dir,'test.art123.csv')
df4 = pd.read_csv(test_fn,index_col=0)

# disabled for now to not alter any more the spreadhseets
# print('python code disabled to not alter the csv files anymore')
# sys.exit()

for XV in range(1,5):
    save_dir = os.path.join(XV_dir,'single_artifact','XV{}'.format(XV))
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for each XV we will rotate the test and valid dataset
    df0,df1,df2,df3,df4 = df1,df2,df3,df4,df0
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



