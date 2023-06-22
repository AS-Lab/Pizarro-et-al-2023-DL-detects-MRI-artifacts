from __future__ import print_function
import os,sys
import nibabel as nib
import random
import time
import scipy.ndimage
import pandas as pd
import numpy as np
np.seterr(all='raise')
from subprocess import check_call



def normalize(img):
    m=np.mean(img)
    st=np.std(img)
    norm = (img - m) / st
    return norm

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
    affine=np.eye(len(np.squeeze(subj_data.shape)))
    if '.mnc.gz' in p:
        img = nib.load(p)
        data = img.get_data()
        data = swap_axes(data)
        # print(data.shape)
        if any(np.asarray(data.shape)>np.asarray(img_shape)):
            data=resize_img(data,img_shape)
        # data = normalize(data)
        data = pad_img(data)
        subj_data = np.reshape(data,input_size)
        return subj_data,img.affine
    else:
        print("File is not mnc.gz : {}".format(p))
        return subj_data,affine


def random_rotate(subj_data):
    input_shape = subj_data.shape
    # maximum angle to define range
    max_angle = 5

    # rotate along (x,y) plane
    alpha = random.uniform(-max_angle,max_angle)
    subj_data = scipy.ndimage.rotate(subj_data, alpha, axes=(0,1),reshape=True)
    # rotate along (x,z) plane
    beta = random.uniform(-max_angle,max_angle)
    subj_data = scipy.ndimage.rotate(subj_data, beta, axes=(0,2),reshape=True)
    # rotate along (y,z) plane
    theta = random.uniform(-max_angle,max_angle)
    subj_data = scipy.ndimage.rotate(subj_data, theta, axes=(1,2),reshape=True)
    # reshape=True above changes the size of the image to we need to resize
    subj_data = resize_img(subj_data,input_shape)
    return subj_data


def random_translation(subj_data):
    input_shape = subj_data.shape
    # wall size will be 5 voxels
    wall = 5
    img_shape_small = tuple(i-wall for i in input_shape[:-1]) + (1,)
    subj_data_small = resize_img(subj_data,img_shape_small)
    corner = tuple([random.randint(0,wall) for i in range(3)])
    subj_data = pad_img(np.squeeze(subj_data_small),c=corner)
    subj_data = np.reshape(subj_data,input_shape)
    return subj_data


def random_flip_xyz(subj_data):
    flip_x = random.randint(0,1)*2-1
    flip_y = random.randint(0,1)*2-1
    flip_z = random.randint(0,1)*2-1
    return subj_data[::flip_x, ::flip_y, ::flip_z, :]

def random_brightness(subj_data):
    amplitude = np.abs(1*random.gauss(0,1))+1
    brighter = random.randint(0,1)
    if not brighter:
        amplitude = 1.0/amplitude
    return amplitude*subj_data



def data_augment(subj_data):
    # random rotation along (x,y), (x,z) and (y,z)
    subj_data = random_rotate(subj_data)
    # random translation in either x, y, and z
    subj_data = random_translation(subj_data)
    # random flip in the x, y, and/or z
    # subj_data = random_flip_xyz(subj_data)
    # amplify by floating number, will brighten if > 1.0 and dim if < 1.0
    # subj_data = random_brightness(subj_data)
    return subj_data



def save_to_nii(data,affine,fn):
    # data = data.reshape(data.shape+(1,))
    # affine=np.eye(len(data.shape))
    img_nii = nib.Nifti1Image(data,affine)
    if not os.path.isfile(fn+'.gz'):
        print('Saving augmented data to gun zipped file : {}'.format(fn+'.gz'))
        nib.save(img_nii,fn)
        check_call(['gzip', fn])
    else:
        print('File {} already exists'.format(fn+'.gz'))




view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# This will have the directory and the label
path = '/home/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098/XV0'
# XV_set = 'XV0'
train_fn = '/home/rpizarro/noise/XValidFns/single_artifact/clean_percent_050/XV0/train.art123.csv'
df_done = pd.read_csv(train_fn,index_col=0)
# valid_fn = os.path.join(path,'valid.art123.csv')
# valid = pd.read_csv(valid_fn,index_col=0)

print(df_done.shape)


# training file to be augmented
fn = '/data/datasets/shared/rpizarro/noise/weights/rap_NN008_multiple_artifact/clean_percent_050/XV0/nb_classes_02/nb_samples_factor_01.00/001-CLR-trg12_NN8mod_priming_ep500/train.art123.ep419.csv'
df = pd.read_csv(fn,index_col=0)

col_list = list(df)
col_list.remove('path')
input_size=(256,256,64,1)

for da_idx in [4]: # range(5):
    print('Going through data augmentation number : {}'.format(da_idx))
    start_da = time.time()

    train_aug_fn = os.path.join(path,'augmented','train.art123.aug{0:03d}.csv'.format(da_idx))
    df_aug = pd.DataFrame(columns=list(df))

    for idx in range(df.shape[0]):
        start_idx = time.time()
        f_orig = df.iloc[[idx]].path.values[0]
        subj_label = df.iloc[[idx]][col_list].values.tolist()[0]
        print("{} : {} : {}".format(idx,subj_label,f_orig))

        f_orig_dir = os.path.dirname(f_orig)

        if f_orig in set(df_done['path']):
            f_aug_dir = f_orig_dir.replace('data/datasets','data2/datasets/aug')
            f_aug_base = os.path.basename(f_orig).replace('.mnc.gz','_aug{0:03d}.nii'.format(da_idx))
            f_aug = os.path.join(f_aug_dir,f_aug_base)
            row = pd.DataFrame([subj_label+[f_aug+'.gz']],columns=list(df))
            df_aug = df_aug.append(row,ignore_index=True)
            print('Skipping since we already augmented : {}'.format(f_orig))
            continue

        print('Could not find so we will augment idx : {}'.format(idx))

        f_aug_dir = f_orig_dir.replace('data/datasets','data/datasets/shared/rpizarro/aug')
        if not os.path.exists(f_aug_dir):
            print('Does not exist so we are creating dir : {}'.format(f_aug_dir))
            os.makedirs(f_aug_dir)
        f_aug_base = os.path.basename(f_orig).replace('.mnc.gz','_aug{0:03d}.nii'.format(da_idx))
        f_aug = os.path.join(f_aug_dir,f_aug_base)
        f_aug_gz = f_aug+'.gz'
        row = pd.DataFrame([subj_label+[f_aug_gz]],columns=list(df))
        df_aug = df_aug.append(row,ignore_index=True)

        if os.path.exists(f_aug_gz):
            print('We have already augmented : {}'.format(f_aug_gz))
            continue
        
        subj_data,affine = get_subj_data(f_orig,input_size)
        subj_data = data_augment(subj_data)  
        save_to_nii(subj_data,affine,f_aug)

        elapsed_idx = time.time() - start_idx
        print('Time it took to augment one file : {0:0.2f} seconds'.format(elapsed_idx))

    elapsed_da = time.time() - start_da
    print('Time it took to augment {0} files : {1:0.2f} minutes'.format(idx,elapsed_da/60.0))

    print('Saving list of augmented files to : {}'.format(train_aug_fn))
    df_aug.to_csv(train_aug_fn)



