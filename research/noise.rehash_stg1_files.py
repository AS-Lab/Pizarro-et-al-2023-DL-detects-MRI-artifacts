import pandas as pd
import os,sys
import glob


def turn_to_csv(f):
    # Using readlines()
    file1 = open(f, 'r')
    Lines = file1.readlines()
    df = pd.DataFrame(columns=['clean','artifact','path'])
    count = 0
    # Strips the newline character
    for line in Lines:
        count += 1
        fn_old,label_str = line.strip().split('***')
        label = eval(label_str)
        # print("Line{}: {}".format(count, line.strip()))
        fn = os.path.join('/trials/data/rpizarro/datasets/',os.path.basename(fn_old))
        if os.path.exists(fn):
            row_append = pd.DataFrame([[label[0],sum(label[1:]),fn]],columns=list(df))
            df = df.append(row_append,ignore_index=True)
            print("Line{}:{}:{}".format(count,fn,label))
            # print('Huzzah!')
            continue
        else:
            print(os.path.basename(fn))
            # print('We need to copy: {}'.format(fn))
    f_csv = f.replace('txt','csv')
    print('Saving to: {}'.format(f_csv))
    df.to_csv(f_csv)

stg1_dir = '/trials/data/rpizarro/noise/XValidFns/stage1_set'

XV_txt_files = glob.glob(os.path.join(stg1_dir,'*art123.txt'))

for f in XV_txt_files:
    # print(f)

    df = turn_to_csv(f)






