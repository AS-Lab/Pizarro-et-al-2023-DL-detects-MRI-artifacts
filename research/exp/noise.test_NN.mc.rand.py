import json
import time
import numpy as np
import nibabel as nib
import scipy.ndimage
import os, csv,sys
import pandas as pd
import glob

from keras.models import model_from_json, load_model

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

def get_rand_data(input_size=(1,256,256,64,1)):
    img = np.random.normal(0,1,input_size)
    img = normalize(img)
    return img

start_time = time.time()

# python noise.train_NN.py rap_NN007_100ep_CLR 500
experiment = sys.argv[1]
mc_runs = int(sys.argv[2])

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# artifacts: 0-geometric distortion, 1-intensity, 2-motion/ringing, 3-coverage, 4-7 other stuff
artifacts_to_model = [1,2,3] # 0-7
paths_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended'
XV_sets = ['test'] #,'valid','train']

model_dir = '/data/datasets/shared/rpizarro/noise/weights/{}/validation_off'.format(experiment)
model_files = glob.glob(os.path.join(model_dir,'model*.h5'))
model_fn = max(model_files, key=os.path.getctime)
print('Loading model : {}'.format(model_fn))
model = load_model(model_fn)

save_dir = '/home/rpizarro/noise/prediction/{}/'.format(experiment)

for XV_set in XV_sets:
    set_fn = os.path.join(paths_dir,'{}.art123.csv'.format(XV_set))
    df = pd.read_csv(set_fn, index_col = 0)
    col_list = list(df)
    col_list.remove('path')
    cols = ['lbl_{}'.format(c) for c in col_list] + ['NN_{}'.format(c) for c in col_list] + ['path {}'.format(XV_set)]
    df_save = pd.DataFrame(columns=cols)

    l = [0.0]*4
    X = get_rand_data()
    f = '/data/datasets/TRIAL/SITE/SUBJECT/VISIT/noise.path.to.file.mnc.gz'
    for i in range(mc_runs):
        out = model.predict(X)[0]
        metric_label_probs_paths = [['{0:0.3f}'.format(i) for i in list(l)] + ['{0:0.3f}'.format(i) for i in list(out)] + [f]]
        row_append = pd.DataFrame(metric_label_probs_paths, columns=cols)
        df_save = df_save.append(row_append, ignore_index=True)
    df_fn = os.path.join(save_dir,'label_prob_rand_noise.csv')
    df_save.to_csv(df_fn)

elapsed_time = time.time() - start_time
print('Amount of time it took to do {0} mc_runs : {1:0.2f} minutes'.format(mc_runs,elapsed_time/60.0))



