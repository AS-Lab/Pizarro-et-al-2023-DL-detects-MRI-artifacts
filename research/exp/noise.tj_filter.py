import pandas as pd
import os,sys

def load_tj_df():
    tj_path = '/home/rpizarro/noise/sheets/tj'
    fn = os.path.join(tj_path,'combined_sets.tj_filter.csv')
    print('Loading file : {}'.format(fn))
    df_keep = pd.read_csv(fn,index_col=0)
    return df_keep

def get_tj_path(csv_fn):
    dn = os.path.dirname(csv_fn)
    bn = os.path.basename(csv_fn)
    bn_tj = bn.replace('.csv','.tj.csv')
    return os.path.join(dn,bn_tj)

# python noise.tj_filter.py path_to_csv_file
csv_fn = sys.argv[1]

if not os.path.exists(csv_fn):
    print('File not found : {}'.format(csv_fn))
    sys.exit()
print('Loading file : {}'.format(csv_fn))
df_pre_tj = pd.read_csv(csv_fn,index_col=0)
if 'path' not in list(df_pre_tj):
    print('column header path missing')
    print('{}'.format(list(df_pre_tj)))
    sys.exit()

df_keep = load_tj_df()
df = df_pre_tj[df_pre_tj.path.isin(df_keep.path)]

csv_fn_tj = get_tj_path(csv_fn)
print('Writing tj_filtered csv to : {}'.format(csv_fn_tj))
df.to_csv(csv_fn_tj)






