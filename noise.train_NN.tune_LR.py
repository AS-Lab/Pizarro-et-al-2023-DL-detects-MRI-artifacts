from __future__ import print_function
import numpy as np
import os
import nibabel as nib
import random
import pickle
import json
import sys
import glob
import scipy.ndimage
import pandas as pd
pd.options.display.width = 0
np.seterr(all='raise')

from keras.models import model_from_json, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.losses import categorical_crossentropy


def get_new_model(NN=7,nb_artifacts=4,verbose=False):
    fn = "../model/NN00{}_mc_{}art.drop.json".format(NN,nb_artifacts)
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(loss=categorical_crossentropy,optimizer='nadam',metrics=['accuracy','mse'])
    # model.compile(loss='mean_squared_logarithmic_error',optimizer='rmsprop',metrics=['accuracy','mse'])
    if verbose:
        print(model.summary())
    return model


def get_model(path='~/weights/rap_NN007/',NN=7,nb_artifacts=4,verbose=False):
    list_of_files = glob.glob(os.path.join(path,'model*.h5'))
    if list_of_files:
        # print(list_of_files)
        model_fn = max(list_of_files, key=os.path.getctime)
        initial_epoch = get_epoch(model_fn)
        print('Loading model on epoch : {} : {}'.format(initial_epoch,model_fn))
        model = load_model(model_fn)
        if verbose:
            print(model.summary())
    else:
        print('We did not find any models.  Getting a new one!')
        model = get_new_model(NN,nb_artifacts,verbose=verbose)
        initial_epoch = 0
    return initial_epoch,model

def get_epoch(path):
    fn = os.path.basename(path)
    parts = fn.split('_vloss_')
    return int(parts[0][-4:])

def get_class_weight(keys=['clean', 'intensity', 'motion', 'coverage'],nb_samples=[18634.0,  106.0,  217.0,  66.0]):
    nb_samples = [float(n) for n in nb_samples]
    nb_classes = len(nb_samples)
    top = max(nb_samples)
    cw = [top/n for n in nb_samples]
    return dict(zip(range(nb_classes),cw))





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



def get_subj_data(p,input_size=(256,256,64,1)):

    img_shape=input_size[:-1]
    subj_data = np.zeros(input_size)
    if '.mnc.gz' in p:
        img = nib.load(p).get_data()  
        img = swap_axes(img)

        if any(np.asarray(img.shape)>np.asarray(img_shape)):
            img=resize_img(img,img_shape)
            # print("Reshaping")

        img = normalize(img)
        img = pad_img(img)
        subj_data = np.reshape(img,input_size)
    else:
        print("File is not mnc.gz : {}".format(p))
    return subj_data


def get_valid_idx(nb_files):
    cutoff = int(0.02*nb_files)
    if random.randint(0,1):
        return random.randint(0,cutoff)
    else:
        return random.randint(cutoff,nb_files)



def fileGenerator(df, valid=False, verbose=False, nb_artifacts=8, nb_step=1, input_size=(256,256,64,1)):
    np.seterr(all='raise')
    X = np.zeros( (nb_step,) + input_size )
    nb_classes=nb_artifacts+1
    Y = np.zeros((nb_step, nb_classes))
    col_list = list(df)
    col_list.remove('path')
    print(col_list)
    while True:
        for idx in range(df.shape[0]):
            n = idx % nb_step
            try:
                # if valid:
                #     idx = get_valid_idx(df.shape[0])
                f = df.iloc[[idx]].path.values[0]
                subj_label = df.iloc[[idx]][col_list].values.tolist()[0]
                if verbose:
                    print("{} : {} : {}".format(idx,subj_label,f))
                subj_data = get_subj_data(f,input_size)
                X[n] = subj_data
                Y[n] = subj_label
            except Exception as e:
                print('\n\n {} : {} : {} : {}\n'.format(str(e),idx,subj_label,f))
                pass
            if n % nb_step == nb_step-1:
                yield X,Y



def get_nb_samples(df,nb_artifacts=3):
    # returns a list nb_samples by summing categorical values
    nb_samples = df.iloc[:,:nb_artifacts+1].sum().tolist()
    return nb_samples

def runNN(experiment,train_df,valid_df,NN,nb_artifacts,nb_step,input_size):
    # The model architecture... layers, parameters, etc... 
    # was defined in modality.save_NNarch_toJson.py
    weights_dir = '/data/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    initial_epoch,model=get_model(weights_dir,NN=NN,nb_artifacts=nb_artifacts,verbose=True)
    nb_samples = get_nb_samples(train_df,nb_artifacts)
    NN_targets = '[clean,intensity,motion,coverage]'
    if nb_artifacts == 1:
        NN_targets = '[clean,artifact]'
    print('The train data has the following nb_samples, {} : {}'.format(NN_targets,nb_samples))   
    # no need to change this for now
    factor_input = '01.00'
    factor = [1.0*float(factor_input)]+nb_artifacts*[1]
    nb_samples = [a*b for a,b in zip(nb_samples,factor)]
    print('Rather than using above nb_samples we will multiple by {} and change to : {}'.format(factor,nb_samples))
    class_weight = get_class_weight(nb_samples=nb_samples)

    nb_samples_valid = get_nb_samples(valid_df,nb_artifacts)
    print('The valid data has the following nb_samples, {} : {}'.format(NN_targets,nb_samples_valid))

    # let's use all the training data to run each epoch
    # fold = 1
    epochs = 200
    for ep in range(epochs):
        # manually set the learning rate
        lr = get_lr(ep,epochs)
        K.set_value(model.optimizer.lr,lr)
        print('We are currently running (epoch,LR) : ({},{})'.format(ep,lr))
        # We will not fold the data because we can use entire train_df
        # train_df = train_df_all.iloc[lambda x: x.index % fold == ep % fold].reset_index(drop=True)
        # Shuffle train_df for each epoch
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        print('with batch size : {}'.format(train_df.shape[0]))
        print(train_df.head())
        h = History()
        checkpath = os.path.join(weights_dir,'model.epoch{0:04d}'.format(ep)+'_loss_{loss:0.2f}.h5')
        checkpointer=ModelCheckpoint(checkpath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        
        steps_per_epoch = int(1.0*train_df.shape[0]/nb_step)
        model.fit_generator(fileGenerator(train_df, valid=False, nb_step=nb_step,verbose=False,nb_artifacts=nb_artifacts,input_size=input_size), class_weight=class_weight, steps_per_epoch=steps_per_epoch, epochs=1, verbose=1,
                callbacks=[h,checkpointer])
        print('Writing saved model to : {}'.format(checkpath))
        # save the performance history for each epoch
        save_history(weights_dir,ep,h)
    # save the weights at the end of epochs
    # model.save('{}/model.FINAL.h5'.format(weights_dir),overwrite=True)

def get_lr(ep,epochs=100):
    lr = np.logspace(-12, 1, epochs)
    return lr[ep]


def save_history(base,ep,h):
    # use json format to save accuracy, loss over epoch
    json_string=json.dumps(h.history)
    fn=os.path.join(base,'history_epoch{0:04d}_parms.json'.format(ep))
    with open(fn, 'w') as outfile:
        json.dump(json_string, outfile)

def group_artifacts(df):
    df['artifact'] = df['intensity'] + df['motion'] + df['coverage']
    df = df[['clean','artifact','path']].reset_index(drop=True)
    return df



# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# change the model file in get_model function

# python noise.train_NN.py rap_NN007_100ep_CLR
experiment = sys.argv[1]

view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())


# This will have the directory and the label
path = '/home/rpizarro/noise/XValidFns/single_artifact/clean_percent_050/XV0'
train_fn = os.path.join(path,'train.art123.csv')
train = pd.read_csv(train_fn,index_col=0)
valid_fn = os.path.join(path,'valid.art123.csv')
valid = pd.read_csv(valid_fn,index_col=0)

train = group_artifacts(train)
valid = group_artifacts(valid)

print('We are using a clean_precent_050 so it could change the LR range')
print('we can tune the learning rate again to find a range to optimize')

print('train:', train.shape,'valid:',valid.shape)

nb_artifacts = train.shape[1]-2

runNN(experiment,train,valid,NN=8,nb_artifacts=nb_artifacts, nb_step=5, input_size=(256,256,64,1))


