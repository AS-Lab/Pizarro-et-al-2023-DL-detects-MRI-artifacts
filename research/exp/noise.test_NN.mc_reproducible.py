import numpy as np
import tensorflow as tf
import random as rn
import json
import time
import nibabel as nib
import scipy.ndimage
import os, csv,sys
import pandas as pd
import glob
from keras.models import model_from_json, load_model


# The below is necessary for starting tensorflow random numbers in a well-defined initial state.
# tf.set_random_seed(42124)
# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
# np.random.seed(42)
# The below is not necessary in this exercise 
# but it can be sure for starting core Python generated random numbers in a well-defined state.
# rn.seed(1234567)


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

def pad_img(img):
    blank=np.zeros((256,256,64))
    blank[:img.shape[0],:img.shape[1],:img.shape[2]]=img
    return blank

def resize_img(img,img_shape):
    size=list(img_shape)
    zoom=[1.0*x/y for x, y in zip(size, img.shape)]
    return scipy.ndimage.zoom(img, zoom=zoom)

def get_subj_data(p,input_size=(1,256,256,64,1)):

    img_shape=input_size[1:-1]
    subj_data = np.zeros(input_size)
    if '.mnc.gz' in p:
        img = nib.load(p).get_data()
        img = swap_axes(img)
        if any(np.asarray(img.shape)>np.asarray(img_shape)):
            img=resize_img(img,img_shape)
        img = normalize(img)
        img = pad_img(img)
        subj_data = np.reshape(img,input_size)
    else:
        print("File is not mnc.gz : {}".format(p))
    return subj_data

def get_mc_start_run(save_dir):
    mc_csv_files = glob.glob(os.path.join(save_dir,'mc*_label_prob*.csv'))
    if not mc_csv_files:
        return 0
    mc_last_file = os.path.basename(max(mc_csv_files, key=os.path.getctime))
    mc_start_run = int(mc_last_file.split('_')[0][2:])
    print('So far we have run {} mc predictions : {}'.format(mc_start_run,mc_last_file))
    return mc_start_run+1

start_time = time.time()

# python noise.train_NN.py rap_NN007_100ep_CLR 500 21 42
experiment = sys.argv[1]
mc_runs = int(sys.argv[2])
tf_seed = int(sys.argv[3])
np_seed = int(sys.argv[4])

if mc_runs==1:
    # Book keeping
    print("Executing:",__file__)
    print("Contents of the file during execution:\n",open(__file__,'r').read())


print('Setting the random seed to (tf,np) : ({},{})'.format(tf_seed,np_seed))

# The below is necessary for starting tensorflow random numbers in a well-defined initial state.
tf.set_random_seed(tf_seed)
# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(np_seed)

# artifacts: 0-geometric distortion, 1-intensity, 2-motion/ringing, 3-coverage, 4-7 other stuff
artifacts_to_model = [1,2,3] # 0-7
paths_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended'
XV_sets = ['test'] #,'valid','train']

model_dir = '/data/datasets/shared/rpizarro/noise/weights/{}/validation_off'.format(experiment)
model_files = glob.glob(os.path.join(model_dir,'model*.h5'))
model_fn = max(model_files, key=os.path.getctime)
print('Loading model : {}'.format(model_fn))
model = load_model(model_fn)

for XV_set in XV_sets:
    save_dir = '/home/rpizarro/noise/prediction/{}/repro/{}/'.format(experiment,XV_set)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    set_fn = os.path.join(paths_dir,'{}.art123.csv'.format(XV_set))
    df = pd.read_csv(set_fn, index_col = 0)
    col_list = list(df)
    col_list.remove('path')
    cols = ['lbl_{}'.format(c) for c in col_list] + ['NN_{}'.format(c) for c in col_list] + ['path {}'.format(XV_set)]

    mc_start_run = get_mc_start_run(save_dir)
    
    for mc in range(mc_start_run,mc_runs):
        print('Working on mc run {}'.format(mc))
        # set_random_seed()
        start_time_mc = time.time()
        df_save = pd.DataFrame(columns=cols)
        for index, row in df.iterrows():
            # if index > 20:
            #     print('We reached 20 paths, lets move to next MC!')
            #     break
            f = row['path']
            l = row[col_list].values.tolist()
            try:

                X = get_subj_data(f)
                out = model.predict(X)[0]
                metric_label_probs_paths = [['{0:0.3f}'.format(i) for i in list(l)] + ['{0:0.3f}'.format(i) for i in list(out)] + [f]]
                row_append = pd.DataFrame(metric_label_probs_paths, columns=cols)
                df_save = df_save.append(row_append, ignore_index=True)
            except Exception as e:
                print('\n Exception \n {} : {} \n'.format(str(e),f))
                pass
        df_fn = os.path.join(save_dir,'mc{0:03d}_tf{1:03d}_np{2:03d}_label_prob_{3}.csv'.format(mc,tf_seed,np_seed,XV_set))
        print('Saving mc run to {}'.format(df_fn))
        df_save.to_csv(df_fn)
        elapsed_time = time.time() - start_time_mc
        print('Amount of time it took to do 1 mc_run : {0:0.2f} minutes'.format(elapsed_time/60.0))

elapsed_time = time.time() - start_time
print('Amount of time it took to do {0} mc_runs : {1:0.2f} minutes'.format(mc_runs,elapsed_time/60.0))



