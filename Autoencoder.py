import os
import PIL
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

from data_augmentation import VolumeAugmentation

dataset_path_AD_ROI = "AD_CTRL/AD_s3"
dataset_path_CTRL_ROI = "AD_CTRL/CTRL_s3"
dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

"""
Import csv metadata
"""

df = pd.read_csv(dataset_path_metadata, sep=',')
head=df.head()
print(head)

#count the entries grouped by the diagnostic group
#Unittest
print(df.groupby('DXGROUP')['ID'].count())

features = ['DXGROUP', 'ID']
print(df[features])

def read_dataset(dataset_path_AD, dataset_path_CTRL, x_id ="AD-", y_id="CTRL-"):
    """
    load images from NIFTI directory
    Parameters
    ----------
    dataset_path_AD: str
        directory path for AD images
    dataset_path_CTRL: str
        directory path for CTRL images
    x_id: str
        identification string in the filename of AD images
    y_id: str
        identification string in the filename of CTRL images

    Returns
    -------
    X : np.array
        array of AD and CTRL images data
    Y: np.array
        array of labels

    fnames_AD: list (?)
        list containig AD images file names
    fnames_CTRL: list (?)
        list containig CTRL images file names

    """
    fnames_AD = glob(os.path.join(dataset_path_AD, f"*{x_id}*.nii"  ))
    fnames_CTRL= glob(os.path.join(dataset_path_CTRL, f"*{y_id}*.nii"  ))
    X = []
    Y = []
    for fname_AD in fnames_AD:
        X.append(nib.load(fname_AD).get_fdata())
        Y.append(1)
    for fname_CTRL in fnames_CTRL:
        X.append(nib.load(fname_CTRL).get_fdata())
        Y.append(0)
    return np.array(X), np.array(Y), fnames_AD, fnames_CTRL

X_o, Y, fnames_AD, fnames_CTRL = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI)

#Da far diventare un Unittest
print(X_o.shape, Y.shape)
print(X_o.min(), X_o.max(), Y.min(), Y.max())

#print(f' {len(fnames_AD)} matching files found: {fnames_AD[0]}, {fnames_AD[1]}, {fnames_AD[2]}, {fnames_AD[3]} ..., {fnames_AD[50]},...,{fnames_AD[-5]}, {fnames_AD[-4]}, {fnames_AD[-3]}, {fnames_AD[-2]}, {fnames_AD[-1]} '  )
#
#Normalization of intensity voxel values
X_o=X_o/X_o.max()


X=X_o[:,36:86,56:106,24:74] #ippocampo
#X=X_o[:,11:109,12:138,24:110] #bordi neri precisi
#X=X_o[:,20:100,20:130,20:100]
#X=X_o
print(X.shape, Y.shape)

"""
Divide the dataset in train, validation and test in a static way

"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#Da fare diventare un Unittest
print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')

'''
Data augmentation
'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=24)
mass_gen = VolumeAugmentation(X_train, Y_train, shape=(X.shape[1], X.shape[2], X.shape[3]))
array_img, labels = mass_gen.augment()
X_train_tot=np.append(X_train, array_img, axis=0)
Y_train_tot=np.append(Y_train, labels, axis=0) #unittest

print(f'X train aug shape: {X_train_tot.shape}, X test shape: {X_test.shape}')
print(f'Y train aug shape: {Y_train_tot.shape}, Y test shape: {Y_test.shape}')

import tensorflow
#from tensorflow.keras.layers import Dropout,Conv3D, Input, Dense, MaxPooling3D, BatchNormalization, ReLU, Flatten, Conv2D, MaxPooling2D, MaxPool3D, GlobalAveragePooling3D
#from tensorflow.keras.models import Model, load_model
from sklearn import metrics

X_train_tot = X_train_tot[:,:,:,:,np.newaxis]
X_test = X_test[:,:,:,:,np.newaxis]

print(f'X train aug shape: {X_train_tot.shape}, X test shape: {X_test.shape}')
print(f'Y train aug shape: {Y_train_tot.shape}, Y test shape: {Y_test.shape}')

'''
Building encoder
'''

def autoencoder(width=128, height=128, depth=64, agg=1, latentDim=16):

    input_data = tensorflow.keras.layers.Input(shape=(width, height, depth, agg))

    encoder = tensorflow.keras.layers.Conv3D(8, (5,5,5), activation='relu')(input_data)
    encoder = tensorflow.keras.layers.MaxPooling3D((2,2,2))(encoder)
    encoder = tensorflow.keras.layers.BatchNormalization()(encoder)

    encoder = tensorflow.keras.layers.Conv3D(16, (3,3,3), activation='relu')(encoder)
    encoder = tensorflow.keras.layers.MaxPooling3D((2,2,2))(encoder)
    encoder = tensorflow.keras.layers.BatchNormalization()(encoder)

    encoder = tensorflow.keras.layers.Conv3D(32, (3,3,3), activation='relu')(encoder)
    encoder = tensorflow.keras.layers.MaxPooling3D((2,2,2))(encoder)
    encoder = tensorflow.keras.layers.BatchNormalization()(encoder)

    VolumeSize=tensorflow.keras.backend.int_shape(encoder)

    encoder = tensorflow.keras.layers.Flatten()(encoder)
    latent = tensorflow.keras.layers.Dense(latentDim)(encoder)

    # Define the encoder model.
    encoder_model = tensorflow.keras.Model(input_data, latent, name="encoder" )

    # Decoder
    decoder_input = tensorflow.keras.layers.Input(shape=(latentDim))
    decoder = tensorflow.keras.layers.Dense(32)(decoder_input)
    decoder = tensorflow.keras.layers.Reshape((1, 1, 1, 32))(decoder)
    decoder = tensorflow.keras.layers.Conv3DTranspose(32, (4,4,4), activation='relu')(decoder)
    decoder = tensorflow.keras.layers.UpSampling3D((2,2,2))(decoder)
    decoder = tensorflow.keras.layers.BatchNormalization()(decoder)

    decoder = tensorflow.keras.layers.Conv3DTranspose(16, (3,3,3), activation='relu')(decoder)
    decoder = tensorflow.keras.layers.UpSampling3D((2,2,2))(decoder)
    decoder = tensorflow.keras.layers.BatchNormalization()(decoder)

    decoder = tensorflow.keras.layers.Conv3DTranspose(8, (3,3,3), activation='relu')(decoder)
    decoder = tensorflow.keras.layers.Conv3DTranspose(8, (2,2,2), activation='relu')(decoder)
    decoder = tensorflow.keras.layers.UpSampling3D((2,2,2))(decoder)
    decoder = tensorflow.keras.layers.BatchNormalization()(decoder)

    decoder = tensorflow.keras.layers.Conv3DTranspose(1, (5,5,5), activation='relu')(decoder)
    decoder = tensorflow.keras.layers.BatchNormalization()(decoder)

    decoder_output= tensorflow.keras.layers.Activation("sigmoid")(decoder)

    # Define the decoder model
    decoder_model = tensorflow.keras.Model(decoder_input, decoder_output, name="decoder")

    #define the autoencoder
    encoded = encoder_model(input_data)
    decoded = decoder_model(encoded)
    autoencoder_model = tensorflow.keras.models.Model(input_data, decoded)

    return autoencoder_model, encoder_model, decoder_model

autoencoder_model, encoder_model, decoder_model=autoencoder(width=X_train_tot.shape[1], height=X_train_tot.shape[2], depth=X_train_tot.shape[3], agg=X_train_tot.shape[4], latentDim=16)
autoencoder_model.summary()


#autoencoder_model.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
autoencoder_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='MAE')
X_train_VAE, X_val_VAE, Y_train_VAE, Y_val_VAE = train_test_split(X_train_tot, Y_train_tot, test_size=0.1)

train_data=X_train_VAE
test_data=X_val_VAE

autoencoder_model.fit(train_data, train_data, epochs=20, batch_size=32, validation_data=(test_data, test_data))
