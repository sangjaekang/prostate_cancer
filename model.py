# -*- coding: utf-8 -*-
%matplotlib inline
import sys, os, re
import time
import random
import shutil

os_path = os.path.abspath('./') 
find_path = re.compile('prostate_cancer')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model

from keras.optimizers import Adam
from keras.utils import np_utils
from keras import initializers
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

from preprocess import *

import h5py

def prediction_first_model(pres_nums,diag_nums,emr_rows,emr_cols,emr_depths,num_classes):
    '''
    model input
        labtest
        1. labtest 중 Max  matrix
        2. labtest 중 Min  matrix
        2. labtest 중 Mean matrix
        4. boolean matrix
        
        prescribe data
        
        diagnosis data
    
    model output
        1. normal - hyper - hypo
    '''
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    pres_input = Input(shape=(pres_nums,emr_cols,1)) 
    diag_input = Input(shape=(diag_nums,emr_cols,1))
    
    # 12 months(long term predicition)
    model_1 = MaxPooling2D((1,12),strides=(1,3),padding='same')(lab_input)
    model_1 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    # 6 months(middle term predicition)
    model_2 = MaxPooling2D((1,6),strides=(1,3),padding='same')(lab_input)
    model_2 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Activation('relu')(model_2)
    # 3 months(short term predicition)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(lab_input)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)
    model_3 = MaxPooling2D((1,3),strides=(1,3),padding='same')(model_3)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_3)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)

    merged = concatenate([model_1,model_2,model_3],axis=1)
    merged = Flatten()(merged)
    merged = Dense(100,activation='relu')(merged)
    
    # prescibtion model
    model_pres = Conv2D(8,(pres_nums,3), padding='same',kernel_initializer=initializers.he_uniform())(pres_input)
    model_pres = BatchNormalization()(model_pres)
    model_pres = Activation('relu')(model_pres)
    model_pres = MaxPooling2D((1,3),strides=(1,3),padding='same')(model_pres)
    model_pres = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_pres)
    model_pres = BatchNormalization()(model_pres)
    model_pres = Activation('relu')(model_pres)
    model_pres = Flatten()(model_pres)
    model_pres = Dense(100,activation='relu')(model_pres)

    # diagnosis data
    model_diag = Conv2D(8,(diag_nums,3), padding='same',kernel_initializer=initializers.he_uniform())(diag_input)
    model_diag = BatchNormalization()(model_diag)
    model_diag = Activation('relu')(model_diag)
    model_diag = MaxPooling2D((1,3),strides=(1,3),padding='same')(model_diag)
    model_diag = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_diag)
    model_diag = BatchNormalization()(model_diag)
    model_diag = Activation('relu')(model_diag)
    model_diag = Flatten()(model_diag)
    model_diag = Dense(100,activation='relu')(model_diag)

    
    merged = concatenate([model_pres,model_diag,merged],axis=1)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)

    out = Dense(num_classes,activation='softmax')(merged)

    model = Model(inputs=[lab_input,diag_input,pres_input], outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def prediction_LABCOMBCNN(pres_nums,diag_nums,emr_rows,emr_cols,emr_depths,num_classes):
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    pres_input = Input(shape=(pres_nums,emr_cols,1)) 
    diag_input = Input(shape=(diag_nums,emr_cols,1))

    vertical_conv_model = Conv2D(emr_rows,(emr_rows,1), padding='same',
                                 kernel_initializer=initializers.he_uniform())(lab_input)
    vertical_conv_model = BatchNormalization()(vertical_conv_model)
    vertical_conv_model = Activation('relu')(vertical_conv_model)
    vertical_conv_model = Conv2D(emr_rows,(emr_rows,1), padding='same',
                                 kernel_initializer=initializers.he_uniform())(vertical_conv_model)
    vertical_conv_model = BatchNormalization()(vertical_conv_model)
    vertical_conv_model = Activation('relu')(vertical_conv_model)
    
    #max_pool_model   = MaxPooling2D((1,3),strides=(1,2),padding='same')(vertical_conv_model)
    temp_conv_model =  Conv2D(32,(emr_rows,3), padding='same',
                              kernel_initializer=initializers.he_uniform())(vertical_conv_model)
    temp_conv_model = BatchNormalization()(temp_conv_model)
    temp_conv_model = Activation('relu')(temp_conv_model)

    # 12 months(long term predicition)
    model_1 = MaxPooling2D((1,12),strides=(1,3),padding='same')(vertical_conv_model)
    model_1 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    # 6 months(middle term predicition)
    model_2 = MaxPooling2D((1,6),strides=(1,3),padding='same')(vertical_conv_model)
    model_2 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Activation('relu')(model_2)
    # 3 months(short term predicition)
    model_3 = MaxPooling2D((1,3),strides=(1,3),padding='same')(vertical_conv_model)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_3)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)
    
    merged = concatenate([model_1,model_2,model_3],axis=1)
    merged = Flatten()(merged)
    merged = Dense(100,activation='relu')(merged)
    
    # prescibtion model
    model_pres = Conv2D(8,(pres_nums,3), padding='same',kernel_initializer=initializers.he_uniform())(pres_input)
    model_pres = BatchNormalization()(model_pres)
    model_pres = Activation('relu')(model_pres)
    model_pres = MaxPooling2D((1,3),strides=(1,3),padding='same')(model_pres)
    model_pres = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_pres)
    model_pres = BatchNormalization()(model_pres)
    model_pres = Activation('relu')(model_pres)
    model_pres = Flatten()(model_pres)
    model_pres = Dense(30,activation='relu')(model_pres)

    # diagnosis data
    model_diag = Conv2D(8,(diag_nums,3), padding='same',kernel_initializer=initializers.he_uniform())(diag_input)
    model_diag = BatchNormalization()(model_diag)
    model_diag = Activation('relu')(model_diag)
    model_diag = MaxPooling2D((1,3),strides=(1,3),padding='same')(model_diag)
    model_diag = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_diag)
    model_diag = BatchNormalization()(model_diag)
    model_diag = Activation('relu')(model_diag)
    model_diag = Flatten()(model_diag)
    model_diag = Dense(30,activation='relu')(model_diag)    
    
    merged = concatenate([model_pres,model_diag,merged],axis=1)
    #fc_model = Flatten()(merged)
    #fc_model = Dropout(0.5)(merged)
    fc_model = Dense(100,activation='relu')(merged)
    #fc_model = Dropout(0.5)(merged)
    fc_model = Dense(100,activation='relu')(fc_model)
    fc_model = BatchNormalization()(fc_model)

    out = Dense(num_classes, activation='softmax')(fc_model)

    model = Model(inputs=[lab_input,diag_input,pres_input], outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

