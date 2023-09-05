import time
import pandas as pd
import os,sys
from subprocess import check_call


aug_dir = '~/noise/XValidFns/single_artifact/clean_percent_050/XV0/augmented'

start_time = time.time()

for aug in range(6,10):
    aug_start = time.time()
    fn = os.path.join(aug_dir,'train.art123.aug{0:03d}.csv'.format(aug))
    df = pd.read_csv(fn)
    print('Working on augmented set {} : {}'.format(aug,fn))
    # print(df['path'])
    for p in df['path']:
        if os.path.exists(p):
            print('Removing : {}'.format(p))
            # check_call(['rm', p])
        else:
            print('File does not exist : {}'.format(p))
    aug_end = time.time() - aug_start
    print('Amount of time it took to remove augmented set : {0:0.2f} minutes'.format(aug_end/60.0))

elapsed_time = time.time() - start_time
print('\nAmount of time it took to remove ALL files : {0:0.2f} minutes\n'.format(aug_end/60.0))



