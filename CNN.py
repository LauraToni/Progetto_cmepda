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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Conv3D, Input, Dense, MaxPooling3D, BatchNormalization, ReLU, Flatten, Conv2D, MaxPooling2D, MaxPool3D, GlobalAveragePooling3D
from tensorflow.keras.models import Model, load_model


try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

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


X=X_o[:,36:86,56:106,24:74]
#X=X_o[:,11:109,12:138,24:110]
#X=X_o
print(X.shape, Y.shape)

"""
Divide the dataset in train , validation and test in a static way

"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

#Da fare diventare un Unittest
print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')



'''
Data augmentation
'''



#augment the dataset



def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(inputs)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = MaxPool3D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=64, activation="relu", kernel_regularizer='l1_l2')(x)
    #x=Flatten()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=X.shape[1], height=X.shape[2], depth=X.shape[3])
model.summary()

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['MAE'])

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "model.{epoch:02d}-{val_MAE:.4f}_C8_16_C32_C64_D64_Hipp.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_MAE", patience=15)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['MAE'])

#And now let's fit it to our data.
#The sample is automatically split in two so that 50% of it is used for validation and the other half for training

history=model.fit(X_train,Y_train, validation_split=0.1, epochs=100, callbacks=[checkpoint_cb, early_stopping_cb])

#history contains information about the training.
#We can now now show the loss vs epoch for both validation and training samples.

print(history.history.keys())
plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])
plt.yscale('log')
plt.show()

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "model.{epoch:02d}-{val_MAE:.4f}_C8_16_C32_C64_D64_Hipp.h5",
    monitor='val_MAE',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', save_freq='epoch')

history = model.fit(X_train,Y_train, validation_split=0.1, epochs=10, callbacks=[checkpoint])

Definiamo la funzione per calcolare l'indice di DICE
def dice(pred, true, k = 1):
    """
    Compute dice coefficient
    Parameters
    ----------
    pred: np.array
        array containing predicted labels
    true:
        array containing true labels
    Returns
    -------
    dice : float
        dice coefficient


    """
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

idx=67
xtrain = X_train[idx][np.newaxis,...]
ytrain = Y_train[idx][np.newaxis,...]
print(Y_train[idx].shape, ytrain.shape)

ypred = model.predict(xtrain).squeeze()>0.1
ytrue = Y_train[idx].squeeze()

dice_value = dice(ypred, ytrue)
print(f'Indice di DICE:{dice_value}')

print(ypred.shape, ytrue.shape)

#use model to predict probability that given y value is 1
y_score = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_score)
#calculate AUC of model
auc = metrics.roc_auc_score(Y_test, y_score)
#print AUC score
print(auc)
#plot roc_curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
