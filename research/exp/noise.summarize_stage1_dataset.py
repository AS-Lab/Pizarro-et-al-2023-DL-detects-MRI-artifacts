
import pandas as pd
import os,sys
import glob

XVset_dir = '/home/rpizarro/noise/XValidFns/29-06-12-19-06.unappended'

XVsets = glob.glob(os.path.join(XVset_dir,'*art123.txt'))

cols = ['clean','intensity','motion','coverage','path']

XVsets_cols = ['XVset']+cols[:-1]
df_XVsets = pd.DataFrame(columns=XVsets_cols)

for XV in XVsets:
    df = pd.DataFrame(columns=cols)
    print('Working on file : {}'.format(XV))
    file1 = open(XV,'r')
    Lines = file1.readlines()
    for line in Lines:
        line_parts = line.strip().split('***')
        path = line_parts[0]
        label = line_parts[1][1:-1].split(',')
        label = [float(l) for l in label]
        # print(label+[path])
        # sys.exit()
        df_line = pd.DataFrame([label + [path]],columns=cols)
        df = df.append(df_line,ignore_index=True)
        # print(label,path)
    # print(df.sum())
    df_sum = list(df.sum())
    print(df_sum[:-1])
    df_set = pd.DataFrame([[os.path.basename(XV).replace('.art123.txt','')]+df_sum[:-1]],columns=XVsets_cols)
    df_XVsets = df_XVsets.append(df_set, ignore_index=True)

df_sum = pd.DataFrame([['Total']+list(df_XVsets.sum())[1:]],columns=XVsets_cols)
df_XVsets = df_XVsets.append(df_sum, ignore_index=True)

print(df_XVsets)
XVsets_summary_fn = os.path.join(XVset_dir,'XVsets_summary.csv')
print('Saving summary of the cross validation sets to : {}'.format(XVsets_summary_fn))
df_XVsets.to_csv(XVsets_summary_fn)




