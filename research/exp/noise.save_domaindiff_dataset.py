import pandas as pd
import glob
import os,sys




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




def make_XVsets(save_dir,noise_clean):

    # This will have the directory and the label
    [train,valid,test]=split_train_valid(noise_clean)


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






art_fn = '/trials/data/rpizarro/noise/XValidFns/multiple_artifact/noise_scans.tj.csv'

art_df = pd.read_csv(art_fn,index_col=0)
art_df = art_df.sample(frac=1).reset_index(drop=True)
art_df['noise'] = art_df['intensity']+art_df['motion']+art_df['coverage']
art_df = art_df[['clean','noise','path']]

df = pd.DataFrame([[0.0,0.0,'path_to_file']],columns=list(art_df))

for index, row in art_df.iterrows():
    fn = os.path.join('/trials/data/rpizarro/datasets',os.path.basename(row['path']))
    if df.shape[0]>=535:
        continue
    if os.path.exists(fn):
        row_append = pd.DataFrame([[row['clean'],row['noise'],fn]],columns=list(art_df))
        df = df.append(row_append,ignore_index=True)



# art_df = art_df.head(535)

clean_files = glob.glob('/trials/data/rpizarro/ricardo_vetted/*.mnc.gz')

for fn in clean_files:
    row_append = pd.DataFrame([[1.0,0.0,fn]],columns=list(art_df))
    df = df.append(row_append,ignore_index=True)
    

print(df)

save_dir = '/trials/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_stg1'


# make_XVsets(save_dir,df)








