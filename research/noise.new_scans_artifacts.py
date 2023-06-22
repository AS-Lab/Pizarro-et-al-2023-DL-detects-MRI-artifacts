import pandas as pd
import os,sys



paths_dir = '/home/rpizarro/noise/XValidFns'

fn = os.path.join(paths_dir,'filtered_failed_scanpaths_20201021.csv')
df = pd.read_csv(fn,index_col=0)
cols = list(df)
# drop Unanmed last four columns
cols = cols[:-4]
df = df[cols]
print(list(df))
# Drop NA from dataframe
df.dropna(subset=['ScanPath'],inplace=True)

fn = os.path.join(paths_dir,'multiple_artifact','noise_scans.tj.csv')
noise = pd.read_csv(fn,index_col=0)
cols_art = ['intensity', 'motion', 'coverage']
noise['max_val'] = noise.max(axis=1)
noise[cols_art] = noise[cols_art].div(noise.max_val, axis=0).round(1)
noise.drop(['max_val'],axis=1,inplace=True)
print(noise)
print(list(noise))

fn = os.path.join(paths_dir,'multiple_artifact','trials_used_noise.tj.csv')
trials_used = pd.read_csv(fn,index_col=0)
print(trials_used.sum())
# Only artifact that is properly identified
keywords = ['motion']
# keywords = ['motion','intensity','artifScanPathact','coverage']

for k in keywords:
    df_k = df[df['Comments'].str.contains(k)]
    df_p = df_k[df_k['ScanPath'].str.contains('mnc.gz')]
    df_p = df_p.drop_duplicates(subset=['ScanPath'])
    print('Using keyword >>>{}<<< results in nb_files : {}'.format(k,df_p.shape[0]))
    # count = df_p[['Trial','Comments']].groupby(['Trial']).agg(['count'])
    count = df_p[['Trial','Comments']].groupby(['Trial']).count()
    for p in df_p['ScanPath'].drop_duplicates():
        p_actual = os.path.join('/data/datasets/FAILEDSCANS',p[1:])
        # print(p_actual)
        if not os.path.exists(p_actual):
            print('We could not find : {}'.format(p_actual))
        elif any(noise['path'].astype(str).str.contains(p_actual)):
            print('\nWe already added : {}\n'.format(p_actual))
            print(noise['path'].astype(str).str.contains(p_actual))
        else:
            print('We will append as movement : {}'.format(p_actual))
            row = pd.DataFrame([[0.0,0.0,1.0,0.0,p_actual]], columns = list(noise))
            noise = noise.append(row,ignore_index=True)
    # print(df_p[['Path','ScanPath']])
    # print(df_k['Comments'])


sys.exit()

print(noise)
fn = os.path.join(paths_dir,'multiple_artifact','noise_scans.tj.new_trials.csv')
print('Saving noise to : {}'.format(fn))
noise.to_csv(fn)

count = count.rename(columns={"Comments": "noise"})

print(list(count))
count = count.reset_index()
count.index = count.index.rename('index')
count.rename(columns={'Trial':'trial'},inplace=True)
print(count)
fn_count = os.path.join(paths_dir,'motion_count_by_trial.csv')
print('Saving count to : {}'.format(fn_count))
# count.to_csv(fn_count)

print(trials_used)
# print(count[['trial','noise']])

trials_used = trials_used.append(count[['trial','noise']],ignore_index=True)

trials_used.sort_values(by=['trial'],inplace=True)
trials_used = trials_used.reset_index(drop=True)

print(trials_used)
fn = os.path.join(paths_dir,'multiple_artifact','trials_used_noise.tj.new_trials.csv')
print('Saving noise to : {}'.format(fn))
trials_used.to_csv(fn)





