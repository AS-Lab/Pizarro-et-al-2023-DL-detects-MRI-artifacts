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

def group_artifacts(df):
    df['artifact'] = df['intensity'] + df['motion'] + df['coverage']
    df = df[['clean','artifact','path']].reset_index(drop=True)
    return df


def load_model_weights(model_dir):
    # model_dir should be weights_dir
    model_fn = '../model/NN007_3art.json'
    with open(model_fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    print(model.summary())
    weights_fn = os.path.join(model_dir,'weights.FINAL.h5')
    model.load_weights(weights_fn)
    return model


# python noise.train_NN.py rap_NN007_100ep_CLR
experiment = sys.argv[1]

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# No. of entries in each class:  [101. 150. 295.  90.]
# clean: 101 # noise: 535 artefact_levels: 3 artefacts_to_model: 123
# train: 477 valid: 127 test: 32

# artifacts: 0-geometric distortion, 1-intensity, 2-motion/ringing, 3-coverage, 4-7 other stuff
artifacts_to_model = [1,2,3] # 0-7

paths_dir = '/home/rpizarro/noise/XValidFns/prob_level_appended'

XV_sets = ['test','valid','train']

model_dir = '/data/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
model = load_model_weights(model_dir)
# model_files = glob.glob(os.path.join(model_dir,'model*.h5'))
# model_fn = max(model_files, key=os.path.getctime)
# model_fn = '/data/datasets/shared/rpizarro/noise/weights/{}/model.FINAL.h5'.format(experiment)
# model = load_model(model_fn)


save_dir = '/home/rpizarro/noise/prediction/{}/prob_level_appended/'.format(experiment)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# epsilon for computing categorical cross entropy (CCE)
eps = 10**-10

for XV_set in XV_sets:
    start_time = time.time()

    print('===We are looking at the {} set==='.format(XV_set))
    set_fn = os.path.join(paths_dir,'{}.art123.csv'.format(XV_set))
    df = pd.read_csv(set_fn, index_col = 0)
    # df = group_artifacts(df)

    col_list = list(df)
    col_list.remove('path')
    print(col_list)
    
    cols = ['MSE','CCE'] + ['lbl_{}'.format(c) for c in col_list] + ['NN_{}'.format(c) for c in col_list] + ['path {}'.format(XV_set)]
    df_save = pd.DataFrame(columns=cols)

    for index, row in df.iterrows():
        f = row['path']
        l = row[col_list].values.tolist()
        print(l,f)
        try:
            start_time_MRI = time.time()
            X = get_subj_data(f)
            out = model.predict(X)[0]
            
            mse = sum([(i-j)**2 for i,j in zip(l,out)])
            cce = sum([-1*(i*np.log(j+eps)+(1-i)*np.log(1-j+eps)) for i,j in zip(l,out)])

            metric_label_probs_paths = [['{0:0.3f}'.format(mse)] + ['{0:0.3f}'.format(cce)] + ['{0:0.3f}'.format(i) for i in list(l)] + ['{0:0.3f}'.format(i) for i in list(out)] + [f]]
            row_append = pd.DataFrame(metric_label_probs_paths, columns=cols)
            df_save = df_save.append(row_append, ignore_index=True)

            # true_line = ['{0}:[{1:0.3f}]'.format(c,li) for (c,li) in zip(col_list,l)]
            # print('==TRUE== ' + '\t'.join(true_line))
            # pred_line = ['{0}:[{1:0.3f}]'.format(c,oi) for (c,oi) in zip(col_list,out)]
            # print('==PRED== ' + '\t'.join(pred_line))
            elapsed_time = time.time() - start_time_MRI
            print('Amount of time it took to test 1 MRI : {0:0.2f} minutes'.format(elapsed_time/60.0))

        except Exception as e:
            print('\n {} : {} \n'.format(str(e),f))
            pass
    print(df_save.head())
    df_fn = os.path.join(save_dir,'label_prob_{}.csv'.format(XV_set))
    df_save.to_csv(df_fn)

    elapsed_time = time.time() - start_time
    print('Amount of time it took to test XVset {0} : {1:0.2f} minutes'.format(XV_set,elapsed_time/60.0))

