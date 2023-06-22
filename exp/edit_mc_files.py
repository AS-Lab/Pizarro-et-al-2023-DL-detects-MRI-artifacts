import pandas as pd
import glob
import os,sys


# this dir no longer exists. we ran this once for a manual fix
csv_file_dir = '/home/rpizarro/noise/prediction/rap_NN007_cw_CLR-003_mc-ommit/train_copy'

csv_files = glob.glob(os.path.join(csv_file_dir,'*.csv'))

print('Already ran this script ... exiting now')
sys.exit()

csv_files.sort()

df_mc000 = pd.read_csv(csv_files[0],index_col=0)
print(csv_files[0])
mc000_len = df_mc000.shape[0]
print(mc000_len)
print(df_mc000.tail())

for fn in csv_files[1:]:
    print(fn)
    df = pd.read_csv(fn, index_col=0)
    df_mc = df.iloc[-mc000_len:].reset_index(drop=True)
    print(df_mc.tail())
    df_mc.to_csv(fn)







