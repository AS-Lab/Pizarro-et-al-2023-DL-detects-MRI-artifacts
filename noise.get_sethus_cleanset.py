import pandas as pd
import os,sys

fn = os.path.join('/trials/data/rpizarro/noise/sheets','dataset_list07-08-07-10-20.csv');

df = pd.read_csv(fn)

# NoiseVector(Geometric,Intensity,Movement,Coverage,Contrast,Acquisition,MissScan,Other),
# ModalityVector(T1C,T1P,T2W,T1G,PDW,FLR,MT_ON,MT_OFF,MTR)
# level 1-not visible, 2-subte, 3-prominent

# We don't use t1g or MTR
mod_options = [0,1,2,4,5,6,7]
df_no_use = df[ (df['level']>1)]
df_use = df[ (df['level']==1) & (df['mod'].isin(mod_options)) & (~df['path'].isin(df_no_use['path']))].reset_index(drop=True)


print(df_use)
print('Modalities : T1C,T1P,T2W,T1G,PDW,FLR,MT_ON,MT_OFF,MTR')
print(df_use['mod'].value_counts())
print('artifact type : Geometric,Intensity,Movement,Coverage,Contrast,Acquisition,MissScan,Other')
print(df_use['artifact'].value_counts())

fn = os.path.join('/trials/data/rpizarro/noise/sheets','stage1_artifact_not_visible.csv')
print('Save to : {}'.format(fn))
df_use.to_csv(fn)

# print(df_use.count())









