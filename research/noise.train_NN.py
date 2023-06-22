from __future__ import print_function
import os,sys
import nibabel as nib
import random
import pickle
import json
import math
import glob
import scipy.ndimage
import pandas as pd
pd.options.display.width = 0
import numpy as np
np.seterr(all='raise')

from keras.models import model_from_json, load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from clr_callback import CyclicLR
# from keras_contrib.layers import InstanceNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from keras.optimizers import Nadam
from keras.utils import np_utils
from keras.losses import categorical_crossentropy, binary_crossentropy
import tensorflow as tf

#TF_GPU_ALLOCATOR=cuda_malloc_async

# tf.config.threading.set_inter_op_parallelism_threads(12)

def get_new_model(NN=7,complexity='minus',mc=True,inst=False,nb_artifacts=4,verbose=False):
    fn = "../model/NN00{}_{}art.drop.json".format(NN,nb_artifacts)
    custom_objects = None
    if mc:
        fn = "../model/NN00{}_mc_{}art.drop.json".format(NN,nb_artifacts)
        # fn = "../model/NN00{}_mc_{}_{}art.drop.json".format(NN,complexity,nb_artifacts)
        if inst:
            fn = "../model/NN00{}_mc_in001_{}art.drop.json".format(NN,nb_artifacts)
            custom_objects = {'InstanceNormalization': InstanceNormalization}
    with open(fn) as json_data:
        d = json.load(json_data)
    print('Will get new model from : {}'.format(fn))
    model = model_from_json(d,custom_objects)
    # lr=1.7e-5 determined emperically with experiments to tune LR
    # nadam = Nadam(lr=1.7e-5)
    model.compile(loss=categorical_crossentropy,optimizer='nadam',metrics=['accuracy','mse'])
    # model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=True, alpha=0.02, from_logits=False),optimizer='nadam',metrics=['accuracy','mse'])
    # model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=False, from_logits=False),optimizer='nadam',metrics=['accuracy','mse'])
    if verbose:
        print(model.summary())
    return model


def get_model(path='~/weights/rap_NN007/',NN=7,complexity='minus',mc=True,inst=False,nb_artifacts=4,verbose=False):
    list_of_files = glob.glob(os.path.join(path,'model.*.h5'))
    if list_of_files:
        model_fn = max(list_of_files, key=os.path.getctime)
        initial_epoch = get_epoch(model_fn)
        print('Loading model on epoch : {} : {}'.format(initial_epoch,model_fn))
        model = load_model(model_fn)
        model.compile(loss=categorical_crossentropy,optimizer='nadam',metrics=['accuracy','mse'])
        model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=False, from_logits=False),optimizer='nadam',metrics=['accuracy','mse'])
        # model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=True, alpha=0.02,from_logits=False),optimizer='nadam',metrics=['accuracy','mse'])
        if verbose:
            print(model.summary())
    else:
        print('We did not find any models.  Getting a new one!')
        model = get_new_model(NN,complexity,mc,inst,nb_artifacts,verbose=verbose)
        initial_epoch = 0
    return initial_epoch,model

def get_epoch(path):
    fn = os.path.basename(path)
    key = 'del.ep'
    if key not in fn:
        print('We could not find key {} in {}'.format(key,path))
        return 0
    parts = fn.split(key)
    return int(parts[1][:4])+1

def get_class_weight(keys=['clean', 'intensity', 'motion', 'coverage'],nb_samples=[18634.0,  106.0,  217.0,  66.0]):
    # default set to 98% clean
    nb_samples = [float(n) for n in nb_samples]
    nb_classes = len(nb_samples)
    top = max(nb_samples)
    cw = [top/n for n in nb_samples]
    print('Using nb_classes {} results in class weights: {}'.format(nb_samples,cw))
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
        img = normalize(img)
        img = pad_img(img)
        subj_data = np.reshape(img,input_size)
    else:
        print("File is not mnc.gz or nii.gz : {}".format(p))
    return subj_data


def random_rotate(subj_data):
    input_shape = subj_data.shape
    # maximum angle to define range
    max_angle = 5

    # rotate along (x,y) plane
    alpha = random.uniform(-max_angle,max_angle)
    subj_data = scipy.ndimage.rotate(subj_data, alpha, axes=(0,1),reshape=True)
    # rotate along (x,z) plane
    beta = random.uniform(-max_angle,max_angle)
    # subj_data = scipy.ndimage.rotate(subj_data, beta, axes=(0,2),reshape=True)
    # rotate along (y,z) plane
    theta = random.uniform(-max_angle,max_angle)
    # subj_data = scipy.ndimage.rotate(subj_data, theta, axes=(1,2),reshape=True)
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
    # we generated an augmented dataset usign random rotation and translation
    # random rotation along (x,y), (x,z) and (y,z)
    # subj_data = random_rotate(subj_data)
    # random translation in either x, y, and z
    # subj_data = random_translation(subj_data)
    # random flip in the x, y, and/or z
    subj_data = random_flip_xyz(subj_data)
    # amplify by floating number, will brighten if > 1.0 and dim if < 1.0
    # subj_data = random_brightness(subj_data)
    return subj_data


def get_valid_idx(nb_files):
    # percentage to have clean MRI scans
    clean_percent = 0.98
    cutoff = int(round((1-clean_percent)*nb_files))
    if random.randint(0,1):
        return random.randint(0,cutoff)
    else:
        return random.randint(cutoff,nb_files)


def fileGenerator(df, valid=False, verbose=False, nb_artifacts=8, nb_step=1, input_size=(256,256,64,1)):
    np.seterr(all='raise')
    X = np.zeros( (nb_step,) + input_size )
    nb_classes = nb_artifacts+1
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
                # datafiles were all copied unstructured into a single directory
                f_path = df.iloc[[idx]].path.values[0]
                f = os.path.join('/trials/data/rpizarro/datasets/',os.path.basename(f_path))
                # regenerated XVsets with valid filename paths for COMBAT 
                # f = df.iloc[[idx]].path.values[0]
                subj_label = df.iloc[[idx]][col_list].values.tolist()[0]
                if verbose:
                    print("{} : {} : {}".format(idx,subj_label,f))
                # put the data in queue : data loader,
                # giuve list of cases, automatically queueing while training
                # will give GPU data ready to go
                subj_data = get_subj_data(f,input_size)
                if not valid:
                    # we will not augment the training data
                    subj_data = data_augment(subj_data)
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


def runNN(experiment,epochs,factor_input,train_fn,valid_fn,NN,complexity,mc,E1_ramp,fine_tune,nb_artifacts,nb_step,input_size):
    # The model architecture... layers, parameters, etc... 
    # was defined in modality.save_NNarch_toJson.py
    # weights_dir = '/data/datasets/shared/rpizarro/noise/weights/{}/'.format(experiment)
    weights_dir = '/trials/data/rpizarro/datasets/shared/rpizarro/noise/weights/{}/'.format(experiment)
    if not os.path.exists(weights_dir):
        print('Creating following experiment dir : {}'.format(weights_dir))
        os.makedirs(weights_dir)
    else:
        print('Weights dir already exists : {}'.format(weights_dir))
    # initial_epoch,model = get_model(weights_dir,NN=NN,complexity=complexity,mc=mc,inst=False,nb_artifacts=nb_artifacts,verbose=True)
    initial_epoch,model = get_model(weights_dir,NN=NN,complexity=complexity,mc=False,inst=False,nb_artifacts=nb_artifacts,verbose=True)
    
    if epochs <= initial_epoch:
        print('We have already trained for {} epochs'.format(initial_epoch))
        sys.exit()
    # keep train_fn and valid_fn unchanged and reload for each epoch
    # train_fn = train_df.fn
    # valid_fn = valid_df.fn
    for ep in range(initial_epoch, epochs):
        # Manually change clean_ratio from 50% to 98% during 500 epochs, then we can gradually incorporate more clean scans # for now set clean_ratio to 0.50
        # clean_ratio = get_clean_ratio(ep=ep)
        data_size,clean_ratio = get_data_size(ep=ep,epochs=epochs,E1_ramp=E1_ramp)
        aug_epochs = 1000
        train_df = load_train_df(train_fn,ep,aug_epochs,nb_artifacts,clean_ratio,data_size)

        nb_samples = get_nb_samples(train_df,nb_artifacts)
        NN_targets = '[clean,intensity,motion,coverage]'
        if nb_artifacts == 1:
            NN_targets = '[clean,artifact]'

        print('The working train data has the following nb_samples, {} : {}'.format(NN_targets,nb_samples))
        factor = [1.0*float(factor_input)]+nb_artifacts*[1]
        nb_samples = [a*b for a,b in zip(nb_samples,factor)]
        print('Rather than using above nb_samples we will multiple by {} and change to : {}'.format(factor,nb_samples))
        # We are not using class weights for FocalLoss
        class_weight = get_class_weight(nb_samples=nb_samples)
        
        # We will keep the validation subset the same at 98% clean to make comparisons across epochs
        valid_df = load_valid_split(valid_fn,nb_classes=nb_artifacts+1,clean_ratio=0.98,ep=ep)
        nb_samples_valid = get_nb_samples(valid_df,nb_artifacts)
        print('The valid data has the following nb_samples, {} : {}'.format(NN_targets,nb_samples_valid))
        
        if ep < E1_ramp:
            # We will save the train and valid sets while ramping between 50%-98% clean
            fn = os.path.join(weights_dir,'train.art123.ep{0:04d}.csv'.format(ep))
            print('The working train set for epoch {} is saved to :\n{}\n'.format(ep,fn))
            train_df.to_csv(fn)
            fn = os.path.join(weights_dir,'valid.art123.ep{0:04d}.csv'.format(ep))
            print('The working validation set for epoch {} is saved to :\n{}\n'.format(ep,fn))
            valid_df.to_csv(fn)
        print('\nWe are currently running epoch {} of {}'.format(ep,epochs-1))
        # random shuffle list
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        print('with train dataset size : {}'.format(train_df.shape[0]))
        print(train_df.head())
        # random shuffle list
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)
        print('with valid dataset size : {}'.format(valid_df.shape[0]))
        print(valid_df.head())
        validation_steps = valid_df.shape[0]//nb_step
        print('\nFor each epoch we will train using the entire set of size : {}'.format(train_df.shape[0]))
        print('After the epoch completes we will validate {} files for this number of steps : {}'.format(nb_step,validation_steps))

        # Callbacks: history, save file, CLR
        h = History()
        checkpath = os.path.join(weights_dir,'model.ep{0:04d}'.format(ep)+'_vloss{val_loss:0.2f}.h5')
        checkpointer = ModelCheckpoint(checkpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        steps_per_epoch = int(1.0*train_df.shape[0]/nb_step)
        base_lr,max_lr = get_lr_range(ep,base_lr=1e-6, max_lr=1e-3,epochs=epochs,fine_tune=fine_tune)
        print('We will use CLR with LR values (base,max) : ({0:0.5e},{1:0.5e})'.format(base_lr,max_lr))
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,step_size=int(round(steps_per_epoch/2.0)))

        # model.fit(fileGenerator(train_df, valid=False, nb_step=nb_step,verbose=False,nb_artifacts=nb_artifacts,input_size=input_size), steps_per_epoch=steps_per_epoch, epochs=1, verbose=1,
        #         validation_data=fileGenerator(valid_df, valid=True, nb_step=nb_step,verbose=False,nb_artifacts=nb_artifacts,input_size=input_size), validation_steps=validation_steps, callbacks=[h,checkpointer])
        model.fit(fileGenerator(train_df, valid=False, nb_step=nb_step,verbose=False,nb_artifacts=nb_artifacts,input_size=input_size),class_weight=class_weight, steps_per_epoch=steps_per_epoch, epochs=1, verbose=1,
                validation_data=fileGenerator(valid_df, valid=True, nb_step=nb_step,verbose=False,nb_artifacts=nb_artifacts,input_size=input_size), validation_steps=validation_steps, callbacks=[h,checkpointer,clr])
        # print('Writing saved model to : {}'.format(checkpath))
        # save the performance history for each epoch
        save_history(weights_dir,ep,h)
    # save the weights at the end of epochs
    # model.save('{0}/model.FINAL.ep{1:04d}.h5'.format(weights_dir,ep),overwrite=True)


def load_train_df(train_fn,ep,aug_epochs,nb_artifacts,clean_ratio,data_size):
    # for 002-CLR-trg12_NN8mod_baseline_ep1500
    # train_df = load_train_clean_ratio(train_fn,nb_classes=nb_artifacts+1,clean_ratio=clean_ratio)
    train_df = load_train_data_size(train_fn,nb_classes=nb_artifacts+1,data_size=data_size)

    # # for 002-CLR-trg12_NN8mod_augmented_ep1500
    # if ep < aug_epochs:
    #     # get augmented set for first 1000 epochs and OG set for last 500 epochs
    #     train_df = load_augmented_set(train_fn,ep)
    # else:
    #     train_df = load_train_clean_ratio(train_fn,nb_classes=nb_artifacts+1,clean_ratio=clean_ratio)
    return train_df


def get_data_size(ep=0,epochs=200,E1_ramp=1):
    # ep is current epochs while epochs is total number of epochs we are training
    # we use epochs to determine the slope of the data ramp
    # For this experiment we will linearly increase clean ratio 50-98 over 500 epochs
    data_size_base = 834
    data_size_top = 20879
    # this value is both the range to ramp the data up to max and the moment when datasize should be maxed
    ep_threshold = -1 # epochs-1
    if E1_ramp > 1:
        ep_threshold = E1_ramp
    nb_artifacts = data_size_base//2
    if ep > ep_threshold:
        data_size = data_size_top
        clean_ratio = 0.98
    else:
        data_size = data_size_base + (ep * (data_size_top - data_size_base) / ep_threshold)
        clean_ratio = float(data_size - nb_artifacts) / (data_size)
    return data_size,clean_ratio


def get_clean_ratio(ep=0):
    # For this experiment we will linearly increase clean ratio 50-98 over 500 epochs
    clean_ratio_base = 0.50
    clean_ratio_top = 0.98
    ep_threshold = -1 # 499
    if ep > ep_threshold:
        return clean_ratio_top
    else:
        return clean_ratio_base + (ep * (clean_ratio_top - clean_ratio_base) / ep_threshold)

def get_lr_range(ep,base_lr=1e-6, max_lr=1e-3,epochs=100,fine_tune=10):
    # Methods for CLR taken from https://github.com/bckenstler/CLR
    # we will implement triangular for the first half of the epochs
    # and then triangular 2 for the second half of the epochs

    # cutoff point : decay after 1000 with augmented data and 250 with original
    # cutoff point for 001-CLR-trg12_NN8mod_ep500 at 250 
    # lets reduce LR range in final fine_tune epochs
    epochs_cutoff = epochs - fine_tune

    # only change max_lr if we are beyond epochs_cutoff
    if ep > epochs_cutoff:
        # epochs range where LR will decay exponentially by alpha
        epochs_decay = epochs - epochs_cutoff
        # alpha definition to decay triangular2
        alpha = math.exp(math.log(base_lr)/float(epochs_decay))
        # implement triangular2 beyond epochs_cutoff
        # amplitude is the range between base and max
        amp = max_lr - base_lr
        # subtract half each time ... or add to base (1/(2^N))*amplitude
        max_lr = base_lr + amp*(alpha**(ep - epochs_cutoff))
    return base_lr,max_lr

def save_history(base,ep,h):
    # use json format to save accuracy, loss over epoch
    json_string=json.dumps(h.history)
    # fn=os.path.join(base,'history_ep{0:04d}_parms.json'.format(ep))
    fn=os.path.join(base,'history_ep{0:04d}_parms.json'.format(ep))
    with open(fn, 'w') as outfile:
        json.dump(json_string, outfile)


def balance_train(train):
    nb_clean = (train['clean']==1).sum()
    nb_artifact = (train['clean']==0).sum()
    print('number of scans that are [clean,artifact] : [{0},{1}]'.format(nb_clean,nb_artifact))
    factor = int(round(nb_clean/nb_artifact))
    print('factor of clean/artifact is {}'.format(factor))
    print('we will simply copy the scans with artifacts this number of times')
    artifact = train[ train['clean']==0 ]
    if factor > 1:
        train = train.append([artifact]*(factor-1),ignore_index=True)
    return train

def deflate_train(train):
    nb_clean = (train['clean']==1).sum()
    nb_artifact = (train['clean']==0).sum()
    print('number of scans that are [clean,artifact] : [{0},{1}]'.format(nb_clean,nb_artifact))
    factor = int(round(nb_clean/nb_artifact))
    print('factor of clean/artifact is {}'.format(factor))
    print('we will simply sample the clean scans to equate the number of artifact scans')
    artifact = train[ train['clean']==0 ]
    clean = train[ train['clean']==1 ]
    clean_sample = clean.sample(n=nb_artifact)
    train = artifact.append(clean_sample,ignore_index=True)
    return train

def group_artifacts(df):
    shape = df.shape
    print('We will group the three artifacts [intensity,motion,coverage] into one category')
    df['artifact'] = df['intensity'] + df['motion'] + df['coverage']
    df = df[['clean','artifact','path']].reset_index(drop=True)
    print('Grouping three artifacts changed shape from {} to {}'.format(shape,df.shape))
    return df

def ommit_train_files(train,ommit):
    train_ommit = train.loc[~train['path'].isin(ommit['path'])].reset_index(drop=True)
    # print(ommit.shape)
    # print(train.shape)
    # print(train_ommit.shape)
    return train_ommit


def get_trials_from_path(df):
    freq = {}
    for f in df['path']:
        trial = f.split('/')[3]
        if trial in freq:
            freq[trial] += 1
        else:
            freq[trial] = 1
    trials_df = pd.DataFrame.from_dict(freq,orient='index',columns=['nb_artifact'])
    return trials_df


def select_by_trails(clean,nb_clean,trials_df):
    factor_mult = nb_clean/trials_df['nb_artifact'].sum()
    trials_df['nb_clean'] = factor_mult*trials_df['nb_artifact']
    trials_df['nb_clean'] = trials_df['nb_clean'].round(0).astype(int)
    clean_df = pd.DataFrame(columns=list(clean))
    freq = {}
    for index,row in clean.iterrows():
        trial = row['path'].split('/')[3]
        if trial not in trials_df.index:
            continue
        if trial not in freq:
            # we have a new trial entry
            clean_df = clean_df.append(row,ignore_index=True)
            freq[trial] = 1
        elif freq[trial] < trials_df['nb_clean'][trial]:
            clean_df = clean_df.append(row,ignore_index=True)
            freq[trial] += 1
        else:
            # Too many from this trial
            continue
        
    return clean_df

def load_train_data_size(train_fn,nb_classes=2,data_size=834):
    train = pd.read_csv(train_fn,index_col=0)
    print('\nThe full train dataset including artifacts and path is size : {}'.format(train.shape))
    if (nb_classes==2) & (train.shape[1]>3):
        train = group_artifacts(train)
    nb_samples = get_nb_samples(train,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    nb_clean = int(data_size - nb_artifact)
    print('We will manually change train set size to {0}, by using size [clean,artifact] : [{1},{2}]'.format(data_size,nb_clean,nb_artifact))

    artifact = train.iloc[:nb_artifact]
    trials_df = get_trials_from_path(artifact)

    clean = train.iloc[nb_artifact:]
    # sort clean to get same scans each time
    clean = clean.sort_values('path').reset_index(drop=True)
    # DO NOT SHUFFLE # clean = clean.sample(frac=1).reset_index(drop=True)
    # clean = clean.head(nb_clean)
    
    # manual hack for new artifact data
    clean = select_by_trails(clean,int(nb_clean/0.8033),trials_df)
    # clean = select_by_trails(clean,int(nb_clean/0.75),trials_df)
    clean_ratio_actual = float(clean.shape[0]) / (artifact.shape[0] + clean.shape[0])
    print('We went through the list of files available and found size [clean,artifact] : [{0},{1}] resulting in actual clean_ratio : {2:0.3f}'.format(clean.shape[0],artifact.shape[0],clean_ratio_actual))

    train = artifact.append(clean,ignore_index=True)
    return train


def load_train_clean_ratio(train_fn,nb_classes=2,clean_ratio=0.5):
    train = pd.read_csv(train_fn,index_col=0)
    print('\nThe full train dataset including artifacts and path is size : {}'.format(train.shape))
    if (nb_classes==2) & (train.shape[1]>3):
        train = group_artifacts(train)
    nb_samples = get_nb_samples(train,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    nb_clean = clean_ratio*nb_artifact/(1 - clean_ratio)
    nb_clean = int(nb_clean)
    print('We will manually change clean_ratio to {0:0.3f}, by using size [clean,artifact] : [{1},{2}]'.format(clean_ratio,nb_clean,nb_artifact))

    artifact = train.iloc[:nb_artifact]
    trials_df = get_trials_from_path(artifact)

    clean = train.iloc[nb_artifact:]
    # sort clean to get same scans each time
    clean = clean.sort_values('path').reset_index(drop=True)
    # DO NOT SHUFFLE # clean = clean.sample(frac=1).reset_index(drop=True)
    # clean = clean.head(nb_clean)
    
    # manual hack for new artifact data
    # clean = select_by_trails(clean,int(nb_clean/0.8033),trials_df)
    clean = select_by_trails(clean,int(nb_clean/0.75),trials_df)
    clean_ratio_actual = float(clean.shape[0]) / (artifact.shape[0] + clean.shape[0])
    print('We went through the list of files available and found size [clean,artifact] : [{0},{1}] resulting in actual clean_ratio : {2:0.3f}'.format(clean.shape[0],artifact.shape[0],clean_ratio_actual))

    train = artifact.append(clean,ignore_index=True)
    return train

def load_valid_split(valid_fn,nb_classes=2,clean_ratio=0.5,ep=0):
    df = pd.read_csv(valid_fn,index_col=0)
    print('\nThe full validation dataset including artifacts and path is size : {}'.format(df.shape))
    df = df.sort_values(by=list(df)).reset_index(drop=True)
    if (nb_classes==2) & (df.shape[1]>3):
        df = group_artifacts(df)
    
    nb_samples = get_nb_samples(df,nb_classes-1)
    nb_artifact = int(nb_samples[-1])
    nb_clean = int(clean_ratio*nb_artifact/(1 - clean_ratio))
    print('We will manually change to corresponding clean_ratio of {0:0.3f}, by using size [clean,artifact] : [{1},{2}]'.format(clean_ratio,nb_clean,nb_artifact))

    artifact = df.iloc[:nb_artifact]
    trials_df = get_trials_from_path(artifact)

    clean = df.iloc[nb_artifact:]
    # sort clean to get same scans each time
    clean = clean.sort_values('path').reset_index(drop=True)
    # DO NOT SHUFFLE # clean = clean.sample(frac=1).reset_index(drop=True)

    # manual hack for new artifact data
    clean = select_by_trails(clean,int(nb_clean/0.8033),trials_df)
    # clean = select_by_trails(clean,int(nb_clean/0.75),trials_df)
    clean_ratio_actual = float(clean.shape[0]) / (artifact.shape[0] + clean.shape[0])
    print('We went through the list of files available and found size [clean,artifact] : [{0},{1}] resulting in actual clean_ratio : {2:0.3f}'.format(clean.shape[0],artifact.shape[0],clean_ratio_actual))
    valid = artifact.append(clean,ignore_index=True)
    
    valid_len = 500
    if (valid.shape[0]>valid_len) & (ep>-1):
        # we are splitting the validation dataset into five parts so that it does not take so long
        split_ratio = 5
        print('Validation set is larger than {} files so we will split into {} folds and for epoch {} we select fold : {}'.format(valid_len,split_ratio,ep,ep % split_ratio))
        valid = valid.iloc[lambda x: x.index % split_ratio == ep % split_ratio].reset_index(drop=True)
    return valid



def load_augmented_data(train):
    train_fn = train.fn
    for aug in range(10):
        train_fn_aug = train_fn.replace('train.art123.csv','augmented/train.art123.aug{0:03d}.csv'.format(aug))
        print('Augmenting training data with files from : {}'.format(train_fn_aug))
        train_aug = pd.read_csv(train_fn_aug,index_col=0)
        train = train.append(train_aug,ignore_index=True)
    return train


def load_augmented_set(train_fn,ep):
    # total number of epochs to train using augmented set
    # aug_epochs = 1000
    
    # number of epochs per augmented set
    epochs_aug = 200
    aug_set = ep//epochs_aug
    train_fn_aug = train_fn.replace('train.art123.csv','augmented/train.art123.aug{0:03d}.csv'.format(aug_set))
    print('For epoch {} we will train with augmented data from : {}'.format(ep,train_fn_aug))
    train = pd.read_csv(train_fn_aug,index_col=0)
    if (nb_classes==2) & (train.shape[1]>3):
        # collapse the three artifacts into a single artifact category
        train = group_artifacts(train)
    # hack to reload next set
    # train.fn = train_fn
    return train

def get_E1_epoch(exp):
    parts = exp.split('E1ramp_')
    E1 = int(parts[1][:3])
    return E1


# python noise.train_NN.py 100 rap_NN007_100ep_CLR/clean_percent_050 XV0 01.00 2 (or 4) 001-initialized_constant_clean_098_ep0050 ramp
epochs = int(sys.argv[1])
# experiment_input = 'rap_NN008_multiple_artifact/clean_percent_stg1' # sys.argv[2]
experiment_input = 'rap_NN008_multiple_artifact/clean_percent_098' # sys.argv[2]
XV_set = 'XV0' # sys.argv[3]
factor = '01.00'  # sys.argv[4]
nb_classes = 2 # int(sys.argv[5])
sub_experiment = sys.argv[2]
ramp = sys.argv[3]
# not being used now : CLR fine_tune epochs
fine_tune = int(sys.argv[4])
# complexity = sys.argv[5]
# InstanceNormalization=True, if false use BatchNormalization
# inst_input = sys.argv[5]
# inst = False

E1_ramp = 1
if ramp=='True':
    E1_ramp = get_E1_epoch(sub_experiment)

view=False

# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

# This will have the directory and the label
path = '/trials/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_098'
# path = '/trials/data/rpizarro/noise/XValidFns/multiple_artifact/clean_percent_stg1'
# path = '/trials/data/rpizarro/noise/XValidFns/stage1_set'
# XV_set = 'XV0'
# train_fn = os.path.join(path,XV_set,'train.art123_combat.csv')
# valid_fn = os.path.join(path,XV_set,'valid.art123_combat.csv')
train_fn = os.path.join(path,XV_set,'train.art123.csv')
valid_fn = os.path.join(path,XV_set,'valid.art123.csv')
# train = pd.read_csv(train_fn,index_col=0)
# we will skip data augmentation for now
# train = load_augmented_data(train)
# valid = pd.read_csv(valid_fn,index_col=0)
# ommit_fn = os.path.join(path,'train.ommit.csv')
# ommit = pd.read_csv(ommit_fn,index_col=0)
# train = ommit_train_files(train,ommit)

# collapse the three artifacts into a single artifact category
# if nb_classes == 2:
#     train = group_artifacts(train)
#     valid = group_artifacts(valid)

# hack to load next set
# train.fn = train_fn
# valid.fn = valid_fn
# print('Rather than using class weights, we will balance the clean to artifact scans manually during each epoch')
# train = balance_train(train)
# print(train)
# sys.exit()

print('We will train with MRIs with multiple artifacts with 50% clean and balanced by artifacts across XVsets')
print('plus: we updated the LR_range and now we continue classifying 2 instead of 4')
print('plus: we were BN on wrong channel... updated to features one')
print('plus: training a new deeper bigger architecture')
print('plus: using data with multiple artifacts')
print('plus: incorporated new data with artifacts from new trials')
print('plus: keeping the validation constant at 98% clean for all epochs')
# print('now: removed CLR, using linear LR for comparison')
print('now: doign some flips for a few epochs')
# print('now: redoing stage 1 with combat')
# print('now: running domain difference for stage 1')
# print('now: using focal loss : loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)')
# print('now: we will ommit using class weights to see if BinaryFocalLoss improves it')
# print('now: Using default parameters')
# print('now: going through N3 to check reproducibility of the experiments')
# print('now: training 001-CLR-trg12_NN8mod_priming_ep500')
# print('now: training 002-CLR-trg12_NN8mod_baseline_ep1500 with 50% clean for comparison')
# print('plus: wrote function to change clean percent. will used 50% for 500 epochs')
# print('now: we will increase data from 800 to 20000 over 200 epochs by incorporating more clean scans')
# print('now: train 003-plateau_clean_098_ep0100 for 100 epochs with 98 percent clean, including a decrease in CLR range over 50-100 epochs')
# print('now: train 001-random_constant_clean_098_ep0050 for 50 epochs with 98 percent clean, including a decrease in CLR range over last 10 epochs')
# print('now: train {0} for {1} epochs with datasize ramp 800-20000 over {2} epochs, including a decrease in CLR range over last {3} epochs'.format(sub_experiment,epochs,E1_ramp,fine_tune))
# print('now: train {0} for {1} epochs with datasize 20000 (no ramp) over {2} epochs, including a decrease in CLR range over last {3} epochs'.format(sub_experiment,epochs,E1_ramp,fine_tune))

# print('now: letting nb_samples_01.00 train for 1500 epochs broken down as follows.')
# print('now: 1500 epochs using the original set, providing baseline comparison for augmented experiment')

# print('now: 100 epochs for each of the 10 augmented set, and 500 epochs original set')
# print('now: comparing two different architectures more and less complex')
# print('plus: testing Batchnormalization versus InstanceNormalization')
# print('plus: update the LR range to cycle bsaed on our tuneLR experiment')
# print('plus: using noise.augment_data.py we generated an augmented dataset of size 10*train data by incorporating random rotation and translation')
# print('we will incorporate additional data augmentation on the fly using random flips and brightness and train for 10,000 epochs')
# print('We will continue to train for another 1000 epochs, in case the first 1k was insufficient')
# print('in addition we will force the class weights either direction a couple of orders of magnitudes')
# print('We will pause the data augmentation for now until we can figure this out first')
# print('train:', train.shape,'valid:',valid.shape)

nb_artifacts = nb_classes - 1

# set mc = monte carlo = True 
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}_{}'.format(factor,complexity))
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),'001-random_constant_clean_098_ep{0:04d}'.format(epochs))
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),'001-initialized_constant_clean_098_ep{0:04d}'.format(epochs))
experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),sub_experiment)
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),'003-plateau_clean_098_ep{0:04d}'.format(epochs))
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),'002-faster_ramp_data_800_to_20000_ep{}'.format(epochs))
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}'.format(factor),'001-CLR-trg12_NN8mod_priming_ep{}'.format(epochs))
# if inst:
#     experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}_in001'.format(factor))
# experiment = os.path.join(experiment_input,XV_set,'nb_classes_{0:02d}'.format(nb_classes),'nb_samples_factor_{}_four_aug'.format(factor))
# set inst to False we are not using instance normalization
# runNN(experiment,factor,train,valid,NN=7,complexity=complexity,mc=True,inst=False,nb_artifacts=nb_artifacts, nb_step=5, input_size=(256,256,64,1))
runNN(experiment,epochs,factor,train_fn,valid_fn,NN=8,complexity='',mc=True,E1_ramp=E1_ramp,fine_tune=fine_tune,nb_artifacts=nb_artifacts, nb_step=5, input_size=(256,256,64,1))


