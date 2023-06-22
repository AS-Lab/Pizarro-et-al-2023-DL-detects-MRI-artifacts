"""
====================================
Define and save NN001 and NN002
====================================

For the modality identification procedure, we explored two Neural Networks.
Here we define and save the two architectures and json strings.  We can
then load them when executing other scripts

"""
print(__doc__)
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn_theano.feature_extraction import fetch_overfeat_weights_and_biases
# from sklearn.feature_extraction import fetch_overfeat_weights_and_biases

import json

from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, Input, Lambda, Add, concatenate 
# from keras_contrib.layers import InstanceNormalization
from keras.optimizers import SGD
import itertools



def getNN002(nb_classes=8,input_shape=(7,256,256,64)):
    
    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial

    # Define the model architecture... layers, parameters, etc...

    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape,name='conv3D_000')) # layer 0
    # 'relu' : rectified linear unit
    model.add(BatchNormalization(axis=1,name='batch_001'))
    model.add(Activation('relu',name='relu_002'))
    model.add(Conv3D(32, (3, 3, 3),padding='same', name='conv3D_003')) # layer 3
    model.add(BatchNormalization(axis=1,name='batch_004'))
    model.add(Activation('relu',name='relu_005')) # layer 5
    model.add(MaxPooling3D(pool_size=(2, 2, 2),name='pool_006'))
    #model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), padding='same',name='conv3D_007')) # layer 7
    model.add(BatchNormalization(axis=1,name='batch_008'))
    model.add(Activation('relu',name='relu_009'))
    model.add(Conv3D(64, (3, 3, 3),padding='same', name='conv3D_010')) # layer 10
    model.add(BatchNormalization(axis=1,name='batch_011'))
    model.add(Activation('relu',name='relu_012'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),name='pool_013'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))

    model.add(Conv3D(128, (3, 3, 3), padding='same',name='conv3D_014')) # layer 14
    model.add(BatchNormalization(axis=1,name='batch_015')) # layer 15
    model.add(Activation('relu',name='relu_016'))
    model.add(Conv3D(128, (3, 3, 3),padding='same', name='conv3D_017')) # layer 17
    model.add(BatchNormalization(axis=1,name='batch_018'))
    model.add(Activation('relu',name='relu_019'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),name='pool_020')) # layer 20
    #model.add(Dropout(0.25))

    model.add(Flatten(name='flat_021'))
    model.add(Dense(512,name='dense_022'))
    model.add(BatchNormalization(axis=1,name='batch_0023'))
    model.add(Activation('relu',name='relu_024'))
    #model.add(Dropout(0.5))    
    model.add(Dense(nb_classes,name='dense_025')) # layer 25

    model.add(Activation('softmax',name='soft_026'))

    # with sgd you instantiate the optimizer before passing it to model.compile()
    # the alternative with default parameters would be: optimizer='sgd'
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model
    



def getNN001(nb_classes=8,input_shape=(7,256,256,64)):
    
    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial

    # Define the model architecture... layers, parameters, etc...

    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape,name='conv3D_000')) # layer 0
    # 'relu' : rectified linear unit
    model.add(BatchNormalization(axis=1,name='batch_001'))
    model.add(Activation('relu',name='relu_002'))
    model.add(Conv3D(32, (3, 3, 3), name='conv3D_003')) # layer 3
    model.add(BatchNormalization(axis=1,name='batch_004'))
    model.add(Activation('relu',name='relu_005')) # layer 5
    model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_006'))
    #model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), padding='same',name='conv3D_007')) # layer 7
    model.add(BatchNormalization(axis=1,name='batch_008'))
    model.add(Activation('relu',name='relu_009'))
    model.add(Conv3D(64, (3, 3, 3),name='conv3D_010')) # layer 10
    model.add(BatchNormalization(axis=1,name='batch_011'))
    model.add(Activation('relu',name='relu_012'))
    model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_013'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))

    model.add(Flatten(name='flat_021'))
    model.add(Dense(256,name='dense_022'))
    model.add(BatchNormalization(axis=1,name='batch_0023'))
    model.add(Activation('relu',name='relu_024'))
    #model.add(Dropout(0.5))    
    model.add(Dense(nb_classes,name='dense_025')) # layer 25

    model.add(Activation('softmax',name='soft_026'))

    # with sgd you instantiate the optimizer before passing it to model.compile()
    # the alternative with default parameters would be: optimizer='sgd'
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model

def getNN003(nb_classes=8,input_shape=(256,256,64,1)):

    # Sethu's LC architecture

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...

    model = Sequential()
    # 'relu' : rectified linear unit
    model.add(Conv3D(8, (12, 12, 3), padding='same', input_shape=input_shape,name='conv3D_003')) # layer 0
    model.add(BatchNormalization(axis=1,name='batch004'))
    model.add(Activation('relu',name='relu_005'))
    model.add(Conv3D(8, (6, 6, 6), padding='same',name='conv3D_006')) # layer 3
    model.add(BatchNormalization(axis=1,name='batch_007'))
    model.add(Activation('relu',name='relu_008'))
    model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_009'))
    model.add(Conv3D(16, (3, 3, 3),name='conv3D_010'))
    model.add(BatchNormalization(axis=1,name='batch_011'))
    model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_012'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Conv3D(16, (3, 3, 3),name='conv3D_013'))
    model.add(BatchNormalization(axis=1,name='batch_014'))
    model.add(Activation('relu',name='relu_015'))
    model.add(MaxPooling3D(pool_size=(4, 4, 1),name='pool_016'))
    model.add(Flatten(name='flat_017'))
    model.add(Dense(nb_classes,name='dense_018'))
    model.add(Activation('softmax',name='soft_019'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model


def getNN004(nb_classes=8,input_shape=(256,256,64,1)):

    # Adapted Sethu's LC architecture

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    pool_size=(4, 4, 4)
    model = Sequential()
    # 'relu' : rectified linear unit
    model.add(Conv3D(8, (12, 12, 3), padding='valid', strides=pool_size,input_shape=input_shape,use_bias=False,name='conv3D_003')) # layer 0
    model.add(BatchNormalization(axis=1,name='batch004'))
    model.add(Activation('relu',name='relu_005'))
    model.add(Conv3D(8, (6, 6, 3), padding='valid',strides=(4, 4, 2), use_bias=False,name='conv3D_006')) # layer 3
    model.add(BatchNormalization(axis=1,name='batch_007'))
    model.add(Activation('relu',name='relu_008'))
    # model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_009'))
    model.add(Conv3D(16, (3, 3, 3),padding='valid',strides=(2,2,1),use_bias=False,name='conv3D_010'))
    model.add(BatchNormalization(axis=1,name='batch_011'))
    # model.add(MaxPooling3D(pool_size=(4, 4, 4),name='pool_012'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Conv3D(16, (3, 3, 3),padding='valid',use_bias=False,name='conv3D_013'))
    model.add(BatchNormalization(axis=1,name='batch_014'))
    model.add(Activation('relu',name='relu_015'))
    # model.add(MaxPooling3D(pool_size=(4, 4, 1),name='pool_016'))
    model.add(Flatten(name='flat_017'))
    model.add(Dense(32,name='dense_018_pre'))
    model.add(Dense(nb_classes,name='dense_018_post'))
    model.add(Activation('softmax',name='soft_019'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model

def getNN005(nb_classes=8,input_shape=(256,256,64,1)):

    # Adapted Sethu's LC architecture

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    model = Sequential()
    # 'relu' : rectified linear unit
    model.add(Conv3D(10, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(20, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(30, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(40, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    # model.add(Dense(50,name='dense_pre'))
    model.add(Dense(nb_classes,name='dense_post'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model

    
def getNN006(nb_classes=8,input_shape=(256,256,64,1)):

    # Adapted Sethu's LC architecture

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    model = Sequential()
    # 'relu' : rectified linear unit
    model.add(Conv3D(10, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(20, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(30, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(40, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dense(50,name='dense_pre'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes,name='dense_post'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model

def getNN007(nb_classes=8,input_shape=(256,256,64,1)):

    # Adapted Sethu's LC architecture

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    model = Sequential()
    # 'relu' : rectified linear unit
    model.add(Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dense(128,name='dense_pre'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,name='dense_post'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    return model

    
def getNN007_mc_bn0(nb_classes=8,input_shape=(256,256,64,1)):

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    inp = Input(input_shape)
    # 'relu' : rectified linear unit
    x = Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape)(inp)
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.25)(x,training=True)
    x = Flatten()(x)
    x = Dense(128,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x,training=True)
    out = Dense(nb_classes,name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model


    
def getNN007_mc(nb_classes=8,input_shape=(256,256,64,1)):

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    inp = Input(input_shape)
    # 'relu' : rectified linear unit
    x = Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape)(inp)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Dropout(0.25)(x,training=True)
    x = Flatten()(x)
    x = Dense(128,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x,training=True)
    out = Dense(nb_classes,name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model


    
def getNN007_mc_minus(nb_classes=8,input_shape=(256,256,64,1)):

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    inp = Input(input_shape)
    # 'relu' : rectified linear unit
    x = Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape)(inp)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Dropout(0.25)(x,training=True)
    x = Flatten()(x)
    x = Dense(64,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x,training=True)
    out = Dense(nb_classes,name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model


    
def getNN007_mc_plus(nb_classes=8,input_shape=(256,256,64,1)):

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    inp = Input(input_shape)
    # 'relu' : rectified linear unit
    x = Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape)(inp)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False)(x)
    x = BatchNormalization(axis=4)(x)
    x = Dropout(0.25)(x,training=True)
    x = Flatten()(x)
    x = Dense(256,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x,training=True)
    out = Dense(nb_classes,name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model


#     
# def getNN007_mc_instance_norm(nb_classes=8,input_shape=(256,256,64,1)):
# 
#     # nb_classes = 8 types of possible artifacts
#     # there are 7 possible number of modalities in each trial
#     # Define the model architecture... layers, parameters, etc...
#     inp = Input(input_shape)
#     # 'relu' : rectified linear unit
#     x = Conv3D(16, (12,12,3), activation='relu', padding='valid', strides=(1,1,1), use_bias=False, input_shape=input_shape)(inp)
#     x = InstanceNormalization(axis=4)(x)
#     x = Conv3D(32, (6, 6, 6), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
#     x = InstanceNormalization(axis=4)(x)
#     x = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,4), use_bias=False)(x)
#     x = InstanceNormalization(axis=4)(x)
#     x = Conv3D(128, (3, 3, 3), activation='relu', padding='valid', strides=(4,4,1), use_bias=False)(x)
#     x = InstanceNormalization(axis=4)(x)
#     x = Dropout(0.25)(x,training=True)
#     x = Flatten()(x)
#     x = Dense(128,name='dense_pre')(x)
#     # x = InstanceNormalization(axis=1)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x,training=True)
#     out = Dense(nb_classes,name='dense_post', activation='softmax')(x)
# 
#     model = Model(inputs=inp, outputs=out)
# 
#     model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
#     print(model.summary())
# 
#     return model
# 

def getNN008(nb_classes=8,input_shape=(256,256,64,1)):

    # nb_classes = 8 types of possible artifacts
    # there are 7 possible number of modalities in each trial
    # Define the model architecture... layers, parameters, etc...
    inp = Input(input_shape)
    # 'relu' : rectified linear unit
    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(inp)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # x = Dropout(0.25)(x,training=True)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # x = Dropout(0.5)(x,training=True)
    x = Dropout(0.5)(x)
    out = Dense(nb_classes,name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model



# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

number_artifact = range(6) #[Geometric, Intensity, Movement, Coverage, Contrast, Acquisition, MissScan, Other]
for NN in [8]:#,3,3.2]: # [1,2]: two NN, NN001, NN002
    for na in number_artifact:
        nm=1 # number of possible modalities
        input_size=(256,256,64,nm)
        # number of classes will be the number of artifacts plus one for clean data
        nb_classes=na+1
        model = getNN008(nb_classes,input_size)
        # save as JSON
        json_string = model.to_json()
        fn = "../model/NN{0:03d}_{1}art.drop.json".format(NN,na)
        print("Saving %s" % fn)
        with open(fn, 'w') as outfile:
            json.dump(json_string, outfile)

        continue
        model_p = getNN007_mc_plus(nb_classes,input_size)
        # save as JSON
        json_string = model_p.to_json()
        fn = "../model/NN{0:03d}_mc_plus_{1}art.drop.json".format(NN,na)
        print("Saving %s" % fn)
        with open(fn, 'w') as outfile:
            json.dump(json_string, outfile)

