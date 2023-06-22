from neuroCombat import neuroCombat
import pandas as pd
import numpy as np
import os,sys
import nibabel as nib
from subprocess import check_call
import scipy.ndimage


def swap_axes(img):
    img = np.swapaxes(img,0,2)
    img = img[::-1, ::-1, :]
    sh=np.asarray(img.shape)
    img = np.swapaxes(img,2,np.argmin(sh))
    return img

def pad_img(img,c=(0,0,0)):
    # c is corner where to locate image
    blank=np.zeros((256,256,64))
    blank[c[0]:c[0]+img.shape[0],c[1]:c[1]+img.shape[1],c[2]:c[2]+img.shape[2]]=img
    return blank

def resize_img(img,img_shape):
    size=list(img_shape)
    zoom=[1.0*x/y for x, y in zip(size, img.shape)]
    return scipy.ndimage.zoom(img, zoom=zoom)

def get_subj_data(p,input_size=(256,256,64,1)):
    img_shape=input_size[:-1]
    subj_data = np.zeros(input_size)
    if '.mnc.gz' in p or '.nii.gz' in p:
        img = np.squeeze(nib.load(p).get_fdata())
        # print(img.shape)
        img = swap_axes(img)
        if any(np.asarray(img.shape)>np.asarray(img_shape)):
            img=resize_img(img,img_shape)
        img = pad_img(img)
        subj_data = np.reshape(img,input_size)
        subj_data = subj_data.flatten()
    else:
        print("File is not mnc.gz or nii.gz : {}".format(p))
    return subj_data


def combine_XV_sets(path):
    train_fn = os.path.join(path,'train.art123.csv')
    df = pd.read_csv(train_fn,index_col=0)
    # valid_fn = os.path.join(path,'valid.art123.csv')
    # df_valid = pd.read_csv(valid_fn,index_col=0)
    test_fn = os.path.join(path,'test.art123.csv')
    df_test = pd.read_csv(test_fn,index_col=0)
    df = df.append(df_test,ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df.sample(frac = 0.33333)
    print('Saving to: {}'.format(os.path.join(path,'train_test.001.csv')))
    df1.to_csv(os.path.join(path,'train_test.001.csv'))
    df2 = df.drop(df1.index).sample(frac = 0.5)
    print('Saving to: {}'.format(os.path.join(path,'train_test.002.csv')))
    df2.to_csv(os.path.join(path,'train_test.002.csv'))
    df3 = df.drop(df1.index).drop(df2.index)
    print('Saving to: {}'.format(os.path.join(path,'train_test.003.csv')))
    df3.to_csv(os.path.join(path,'train_test.003.csv'))
    # df = df.append(df_valid,ignore_index=True)
    return df


def get_data(subset_fn):
    # train_fn = os.path.join(path,'train.art123.csv')
    df = pd.read_csv(subset_fn)
    print(df.sum())
    print(df)
    nb_clean = df['clean'].sum()
    nb_artifact = df['artifact'].sum()
    print(df['path'].head(5))
    data = np.zeros((4194304,df.shape[0]))
    data_files = []
    clean = 0
    for fn in df[df['clean']==1.0]['path']:
        if clean>nb_clean:
            break
        print(clean)
        print(fn)
        subj_data = get_subj_data(fn)
        print(subj_data.shape)
        data[:,clean] = subj_data
        data_files += [fn]
        clean = clean+1
    noise = 0
    for fn in df[df['artifact']==1.0]['path']:
        if noise>nb_artifact:
            break
        print(noise)
        print(fn)
        subj_data = get_subj_data(fn)
        print(subj_data.shape)
        data[:,noise+clean] = subj_data
        data_files += [fn]
        noise = noise+1
    return data,data_files,nb_clean,nb_artifact

def save_to_nii(data,fn):
    # data = data.reshape(data.shape+(1,))
    affine=np.eye(len(data.shape))
    img_nii = nib.Nifti1Image(data,affine)
    path=fn
    print(path)
    if not os.path.isfile(path+'.gz'):
        nib.save(img_nii,path)
        print('Saving and gzipping to: {}'.format(path+'.gz'))
        check_call(['gzip', path])
    else:
        print('File {} already exists'.format(path+'.gz'))

def sort_and_save(data_combat,data_files):
    for idx,fn in enumerate(data_files):
        subj_data_combat_Nx1 = data_combat[:,idx]
        subj_data_combat = subj_data_combat_Nx1.reshape((256,256,64,1))
        fn_combat_nii = fn.replace('.mnc.gz','_combat.nii').replace('datasets','combat')
        save_to_nii(subj_data_combat,fn_combat_nii)



def combat_subset(path,idx):
    subset_fn = os.path.join(path,'train_test.00{}.csv'.format(idx+1))
    print('Working with: {}'.format(subset_fn))
    
    data,data_files,nb_clean,nb_artifact = get_data(subset_fn)
    print(data)
    print(data.shape)
    
    print(data_files)
    print(nb_clean,nb_artifact)
    study = int(nb_clean)*[1]+int(nb_artifact)*[2]
    print(study)
    # Specifying the batch (scanner variable) as well as a biological covariate to preserve:
    covars = {'study':study}
    # covars = {'study':[1,1,1,1,1,2,2,2,2,2]}
    covars = pd.DataFrame(covars)  
    
    # To specify names of the variables that are categorical:
    # categorical_cols = ['gender']
    
    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = 'study'
    
    #Harmonization step:
    data_combat = neuroCombat(dat=data,
        covars=covars,
        batch_col=batch_col)["data"]
    
    
    print(data_combat.shape)
    
    
    sort_and_save(data_combat,data_files)
    
    


# Get data
# 200 rows (features) and 10 columns (scans)
# data = np.genfromtxt('testdata/testdata.csv', delimiter=",", skip_header=1)

path = '/trials/data/rpizarro/noise/XValidFns/stage1_set/XV0'
for i in range(3):
    combat_subset(path,i)


