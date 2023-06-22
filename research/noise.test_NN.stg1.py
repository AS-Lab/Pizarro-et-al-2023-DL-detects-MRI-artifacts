import json
import time
import numpy as np
import nibabel as nib
import scipy.ndimage
import os, csv,sys
import pandas as pd
import glob

from keras.models import model_from_json, load_model
from keras.losses import categorical_crossentropy, binary_crossentropy
# from keras_contrib.layers import InstanceNormalization

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

def flip_img(img,mc):
    # there are eight orientations
    orientation = mc % 8
    if orientation==0:
        print('We are in orientation: {}'.format(orientation))
        # 000
        img = img[:,:,:]
    if orientation==1:
        # 001
        img = img[:,:,::-1]
    if orientation==2:
        # 010
        img = img[:,::-1,:]
    if orientation==3:
        # 011
        img = img[:,::-1,::-1]
    if orientation==4:
        # 100
        img = img[::-1,:,:]
    if orientation==5:
        # 101
        img = img[::-1,:,::-1]
    if orientation==6:
        #110
        img = img[::-1,::-1,:]
    if orientation==7:
        #111
        img = img[::-1,::-1,::-1]
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


def get_subj_data(p,mc=1,input_size=(256,256,64,1)):
    img_shape=input_size[:-1]
    subj_data = np.zeros(input_size)
    if '.mnc.gz' in p or '.nii.gz' in p:
        img = np.squeeze(nib.load(p).get_fdata())
        print(img.shape)
        img = swap_axes(img)
        print(img.shape)
        if any(np.asarray(img.shape)>np.asarray(img_shape)):
            img=resize_img(img,img_shape)
        img = normalize(img)
        print(img.shape)
        img = pad_img(img)
        print(img.shape)
        subj_data = np.reshape(img,(1,)+input_size)
        print(subj_data.shape)
    else:
        print("File is not mnc.gz or nii.gz : {}".format(p))
    return subj_data

def get_mc_start_run(save_dir):
    mc_csv_files = glob.glob(os.path.join(save_dir,'mc*_label_prob*.csv'))
    if not mc_csv_files:
        print('So far we have not run any mc predictions')
        return 0
    mc_last_file = os.path.basename(max(mc_csv_files, key=os.path.getctime))
    mc_start_run = int(mc_last_file.split('_')[0][2:])
    print('So far we have run {} mc predictions : {}'.format(mc_start_run,mc_last_file))
    return mc_start_run+1

def group_artifacts(df):
    df['artifact'] = df['intensity'] + df['motion'] + df['coverage']
    df = df[['clean','artifact','path']].reset_index(drop=True)
    return df

def get_ep_max(fn):
    fn_parts = fn.split('ep')
    ep_part = fn_parts[1].split('_')[0]
    return int(ep_part)


def get_models_fn(exp_dirs,model_epochs=[-1]):

    experiment = '/'.join(exp_dirs)
    model_dir = '/trials/data/rpizarro/datasets/shared/rpizarro/noise/weights/{}'.format(experiment)
    print('Looking for models under dir : {}'.format(model_dir))
    model_files = glob.glob(os.path.join(model_dir,'model.ep*.h5'))
    model_files = sorted(model_files, key=os.path.getctime)
    
    ep_max = get_ep_max(os.path.basename(model_files[-1]))

    models_fn = [model_files[idx] for idx in model_epochs if idx<=ep_max]
    return models_fn

def get_epoch(model_fn):
    fn_parts = model_fn.split('ep')
    ep = fn_parts[1].split('_')
    return int(ep[0])


def get_nb_samples(df,nb_artifacts=3):
    # returns a list nb_samples by summing categorical values
    nb_samples = df.iloc[:,:nb_artifacts+1].sum().tolist()
    return nb_samples


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



def load_df(fn,nb_classes=2,clean_ratio=0.5):
    print('Loading files from : {}'.format(fn))
    df = pd.read_csv(fn, index_col = 0)
    if 0: # clean_ratio<0.98:
        nb_samples = get_nb_samples(df,nb_classes-1)
        nb_artifact = int(nb_samples[-1])

        nb_clean = clean_ratio*nb_artifact/(1 - clean_ratio)
        nb_clean = int(nb_clean)
        print('We will manually change clean_ratio to {0:0.3f}, by using size [clean,artifact] : [{1},{2}]'.format(clean_ratio,nb_clean,nb_artifact))

        artifact = df.iloc[:nb_artifact]
        trials_df = get_trials_from_path(artifact)

        clean = df.iloc[nb_artifact:]
        # sort clean to get same scans each time
        clean = clean.sort_values('path').reset_index(drop=True)

        # manual hack for new artifact data
        clean = select_by_trails(clean,int(nb_clean/0.8),trials_df)
        clean_ratio_actual = float(clean.shape[0]) / (artifact.shape[0] + clean.shape[0])
        print('We went through the list of files available and found size [clean,artifact] : [{0},{1}] resulting in actual clean_ratio : {2:0.3f}'.format(clean.shape[0],artifact.shape[0],clean_ratio_actual))

        df = artifact.append(clean,ignore_index=True)
    return df


def get_new_model(NN=8,complexity='minus',inst=False,nb_artifacts=1,verbose=False):
    fn = "../model/NN00{}_{}art.drop.json".format(NN,nb_artifacts)
    with open(fn) as json_data:
        d = json.load(json_data)
    print('Will get new model from : {}'.format(fn))
    model = model_from_json(d,custom_objects=None)
    # lr=1.7e-5 determined emperically with experiments to tune LR
    # nadam = Nadam(lr=1.7e-5)
    model.compile(loss=binary_crossentropy,optimizer='nadam',metrics=['accuracy','mse'])
    # model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=True, alpha=0.02, from_logits=False),optimizer='nadam',metrics=['accuracy','mse'])
    if verbose:
        print(model.summary())
    return model



def run_NN_test_mc(models_fn,exp_dirs,XV_sets,mc_runs,longitudinal='False',cln_prct='clean_percent_098'):
    experiment = '/'.join(exp_dirs)
    paths_dir = '/trials/data/rpizarro/noise/XValidFns/stage1_set/XV0'

    for model_fn in models_fn:

        start_time_model = time.time()

        print('Loading model : {}'.format(model_fn))
        custom_objects = None
        model = load_model(model_fn,custom_objects)
        print(model.summary())
        ep = get_ep_max(os.path.basename(model_fn))
        print('The model was saved after epoch : {}'.format(ep))

        for XV_set in XV_sets:
            # save_dir = '/home/rpizarro/noise/prediction/{0}/epoch_transience/ep{1:04d}/{2}/'.format(experiment,ep+1,XV_set)
            save_dir = '/trials/data/rpizarro/noise/prediction/{0}/{1}/'.format(experiment,XV_set)
            print('We will write the MC files to : {}'.format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            # use the constant reference for comparison, regardless of schemes used in training
            set_fn = os.path.join(paths_dir,'{0}.art123_combat.csv'.format(XV_set))
            df = load_df(set_fn,clean_ratio=0.98)
            if ('nb_classes_02' in experiment) & (df.shape[1]>3):
                print('Number of classes is 2, we will group artifacts')
                # nb_classes = 2
                df = group_artifacts(df)
            else:
                print('We did not find nb_classes_02 in {}'.format(experiment))
                print('Or the XV_set is already in correct shape : {}'.format(df.shape))
            col_list = list(df)
            col_list.remove('path')
            cols = ['lbl_{}'.format(c) for c in col_list] + ['NN_{}'.format(c) for c in col_list] + ['path {}'.format(XV_set)]

            mc_start_run = get_mc_start_run(save_dir)
            
            for mc in range(mc_start_run,mc_runs):
                print('Working on mc run {} of {}'.format(mc,mc_runs))
                start_time_mc = time.time()
                df_save = pd.DataFrame(columns=cols)
                for index, row in df.iterrows():
                    f = row['path']
                    # f = os.path.join('/trials/data/rpizarro/datasets/',os.path.basename(f_path))
                    l = row[col_list].values.tolist()
                    print('Working on file : {}'.format(f))
                    try:
                        X = get_subj_data(f,mc)
                        print(X.shape)
                        out = model.predict(X)[0]
                        metric_label_probs_paths = [['{0:0.3f}'.format(i) for i in list(l)] + ['{0:0.3f}'.format(i) for i in list(out)] + [f]]
                        row_append = pd.DataFrame(metric_label_probs_paths, columns=cols)
                        df_save = df_save.append(row_append, ignore_index=True)
                    except Exception as e:
                        print('\n Exception \n {} : {} \n'.format(str(e),f))
                        pass
                df_fn = os.path.join(save_dir,'mc{0:03d}_label_prob_{1}.csv'.format(mc,XV_set))
                print('Writing prediction for mc run {} : {}'.format(mc,df_fn))
                if mc>1:
                    # drop paths column, already saved in mc000
                    df_save = df_save.drop(['path {}'.format(XV_set)],axis=1)
                df_save.to_csv(df_fn)
                elapsed_time = time.time() - start_time_mc
                print('Amount of time it took to do 1 mc_run : {0:0.2f} minutes'.format(elapsed_time/60.0))

        elapsed_time = time.time() - start_time_model
        print('Amount of time it took to do {0} mc_runs : {1:0.2f} minutes'.format(mc_runs,elapsed_time/60.0))





# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())


start_time = time.time()

# python noise.train_NN.py 500 rap_NN007_100ep_CLR XV0 nb_classes_02 nb_samples_factor_01.00
mc_runs = int(sys.argv[1])
sub_exp = sys.argv[2]
longitudinal = sys.argv[3]
cln_prct = sys.argv[4]
XV_sets = [sys.argv[-1]] #,'valid','train']

exp_dirs = ['rap_NN008_multiple_artifact/{}'.format(cln_prct), 'XV0', 'nb_classes_02', 'nb_samples_factor_01.00', sub_exp]

# specify epochs to select models
# model_epochs = [0]+list(range(9,1000,10))
# model_epochs = [1499] # list(range(879,1000,10))
# select final model
model_epochs = [-1]
if longitudinal == 'True':
    model_epochs = list(range(50))
# 002-ramp_clean_050_to_098_ep0500
# model_epochs = [0,200,400,420,440,450,460,480,485,499]

models_fn = get_models_fn(exp_dirs,model_epochs)



run_NN_test_mc(models_fn,exp_dirs,XV_sets,mc_runs,longitudinal,cln_prct)

elapsed_time = time.time() - start_time
print('Amount of time it took to test all sets : {0:0.2f} minutes'.format(elapsed_time/60.0))

