import os
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

from data_augmentation import VolumeAugmentation
from input_dati import read_dataset,import_csv, cut_file_name
from statistics import roc_curve

#Attivare il comando sottostante per utilizzare plaidml
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#pylint: disable=invalid-name
#pylint: disable=line-too-long

def normalize(x):
    """
    Normalize the intensity of every pixel in the image

    Parameters
    ----------
    x : 4D np.array
        array containing the images
    Returns
    -------
    x : 4D np.array
        array containg the normalized images

    """
    return x/x.max()

def stack_train_augmentation(img, img_aug, lbs, lbs_aug):
    """
    Creates an array containing both original and augmented images. Does the same with their labels

    Parameters
    ----------
    img : 4D np.array
        array containing the images used for the training
    img_aug: 4D np.array
        array containing the augmented images used for the training
    lbs: np.array
        array containing the original image labels
    lbs_aug: np.array
        array containing the augmented image labels
    Returns
    -------
    img_tot : np.array
        array cointaing both original and augmented images
    lbs_tot : np.array
        array containing original and augmented image labels

    """
    img_tot=np.append(img, img_aug, axis=0)
    lbs_tot=np.append(lbs, lbs_aug, axis=0)
    return img_tot, lbs_tot

def get_model(width=128, height=128, depth=64):
    """
    Built a 3D CNN model.

    Parameters
    ----------
    widht: int
        first image's dimension
    height: int
        second image's dimension
    depth: int
        third image's dimension

    Returns
    -------
    model: tensorflow.keras.model()
        the model of the CNN

    """

    inputs = tensorflow.keras.Input((width, height, depth, 1))

    x = tensorflow.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu",kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-3))(inputs)
    x= tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-3))(x)
    x = tensorflow.keras.layers.BatchNormalization(axis=2)(x)
    x= tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-3))(x)
    x= tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-3))(x)
    x = tensorflow.keras.layers.BatchNormalization(axis=2)(x)
    x= tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tensorflow.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu",kernel_regularizer=tensorflow.keras.regularizers.l2(l2=1e-3))(x)
    x = tensorflow.keras.layers.BatchNormalization(axis=2)(x)
    x= tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dropout(0.1)(x)
    outputs = tensorflow.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tensorflow.keras.Model(inputs, outputs, name="3dcnn")
    return model


if __name__=='__main__':
    dataset_path_AD_ROI = "AD_CTRL/AD_ROI"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head, dic_info = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    # import images, labels and file names
    X_o, Y, fnames_AD, fnames_CTRL, file_id, file_age = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI, dic_info, str_1='1', str_2='.')
    #Normalization of intensity voxel values
    X_o=normalize(X_o)

    # Define ROI
    #X=X_o[:,36:86,56:106,24:74] #ippocampo
    #X=X_o[:,11:109,12:138,24:110] #bordi neri precisi
    #X=X_o[:,20:100,20:130,20:100]
    X=X_o

    # Divide the dataset in train, validation and test in a static way
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=11)

    #Augment the data using VolumeAugmentation class
    mass_gen = VolumeAugmentation(X_train, Y_train, shape=(X.shape[1], X.shape[2], X.shape[3]))
    array_img, labels = mass_gen.augment()

    # Create an array containing both original and augmented data
    X_train_tot, Y_train_tot=stack_train_augmentation(X_train, array_img, Y_train, labels)

    # Augement the images of one dimension
    X_train_tot = tensorflow.expand_dims(X_train_tot, axis=-1)
    X_test = tensorflow.expand_dims(X_test, axis=-1)

    # Build the model
    model = get_model(width=X.shape[1], height=X.shape[2], depth=X.shape[3])
    model.summary()

    # Set the learning Rate
    initial_learning_rate = 0.001
    ReduceLROnPlateau=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['MAE'])

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "3d_image_classification_{val_MAE}_Hipp.h5", save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    #Fit the data
    history=model.fit(X_train_tot,Y_train_tot, validation_split=0.1, batch_size=32, shuffle=TRUE, epochs=60, callbacks=[early_stopping, ReduceLROnPlateau])

    #history contains information about the training
    print(history.history.keys())

    #Show the loss vs epoch for both validation and training samples.
    plt.plot(history.history["val_loss"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["MAE"])
    plt.plot(history.history["val_MAE"])
    plt.legend()
    plt.yscale('log')
    plt.show()

    #Display ROC curve and calculate AUC
    auc = roc_curve(X_test, Y_test)
