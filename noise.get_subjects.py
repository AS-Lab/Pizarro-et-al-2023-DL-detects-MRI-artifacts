import pandas as pd
pd.options.display.max_colwidth = 200

def get_study_site(df):
    paths = df['path']
    # print(paths)
    study_site = [p.replace('/data/datasets/','') for p in paths]
    study_site = ['_'.join(ss.split('/')[:2]) for ss in study_site]
    return study_site

def get_ss_subj(df):
    paths = df['path']
    # print(paths)
    study_site_subj = [p.replace('/data/datasets/','') for p in paths]
    study_site_subj = ['_'.join(ss.split('/')[:3]) for ss in study_site_subj]
    return study_site_subj



fn = '/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/train.art123.csv'
df_train = pd.read_csv(fn)

fn = '/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/test.art123.csv'
df_test = pd.read_csv(fn)

fn = '/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098/XV0/valid.art123.csv'
df_valid = pd.read_csv(fn)

print(df_train.loc[[100]]['path'])



study_site_train = get_study_site(df_train)
study_site_test = get_study_site(df_test)
study_site_valid = get_study_site(df_valid)

study_site = study_site_train+study_site_test+study_site_valid

df_ss_repeats = pd.DataFrame(study_site)
print(df_ss_repeats)

df_ss = pd.DataFrame(set(study_site))
print(df_ss)


ss_subj_train = get_ss_subj(df_train)
ss_subj_test = get_ss_subj(df_test)
ss_subj_valid = get_ss_subj(df_valid)

ss_subj = ss_subj_train+ss_subj_test+ss_subj_valid

df_sss_repeats = pd.DataFrame(ss_subj)
print(df_sss_repeats)

df_sss = pd.DataFrame(set(ss_subj))
print(df_sss)


