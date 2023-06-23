import pandas as pd
import os,sys
import glob

XV_path = '/trials/data/rpizarro/noise/XValidFns/stage1_set/XV0/'

XV_fns = glob.glob(os.path.join(XV_path,'*art123.csv'))

for XV in XV_fns:
    print('Working on file: {}'.format(XV))
    df = pd.read_csv(XV,index_col=0)
    print(df)
    df['path']=df['path'].str.replace('datasets','combat')
    df['path']=df['path'].str.replace('.mnc.gz','_combat.nii.gz',regex=False)
    XV_combat = XV.replace('.csv','_combat.csv')
    print('Saving to: {}'.format(XV_combat))
    df.to_csv(XV_combat)








