""" Convolutional neural network applied to neurological AD and CTRL images
to predict features with trasfer learning"""
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
from CNN import normalize, stack_train_augmentation
from input_dati import read_dataset,import_csv, cut_file_name
from statistics import roc_curve, plot_cv_roc

def training_tl(X, feature, size):
    """
    Train the transfer learning model

    :Parameters:
        X : 4D np.array
            Images
        feature : 1D np.array
            labels of the feature considered
        size : float
            Percentage of dataset used as test set
    :Returns:
        None
    """

    # Divide the dataset in train, validation and test in a static way
    X_train, X_test, Y_train, Y_test = train_test_split(X, feature, test_size=size, random_state=11)

    #Augment the data using VolumeAugmentation class
    mass_gen = VolumeAugmentation(X_train, Y_train, shape=(X.shape[1], X.shape[2], X.shape[3]))
    array_img, labels = mass_gen.augment()

    # Create an array containing both original and augmented data
    X_train_tot, Y_train_tot=stack_train_augmentation(X_train, array_img, Y_train, labels)

    # Augement the images of one dimension
    X_train_tot = tf.expand_dims(X_train_tot, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)

    # Define the neural network

    input1 = tf.keras.Input(shape=(50, 50, 50, 1))

    conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(input1)
    relu1 = tf.keras.layers.ReLU()(conv1)
    conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(relu1)
    bat1 = tf.keras.layers.BatchNormalization(axis=2)(conv2)
    relu2= tf.keras.layers.ReLU()(bat1)
    max1 = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(relu2)

    conv3 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(max1)
    relu3= tf.keras.layers.ReLU()(conv3)
    conv4 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(relu3)
    bat2 = tf.keras.layers.BatchNormalization(axis=2)(conv4)
    relu4= tf.keras.layers.ReLU()(bat2)
    max2 = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(relu4)

    conv5 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(max2)
    bat3 = tf.keras.layers.BatchNormalization(axis=2)(conv5)
    relu5= tf.keras.layers.ReLU()(bat3)
    max3= tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(relu5)

    # Define the new model
    base_model = tf.keras.Model(input1, max3, name="3dcnn")

    # Build the base model
    base_model.summary()
    base_model.load_weights('Modelli/CNN_weights_15_50_100.h5', by_name=True)
    base_model.trainable = False

    # Set the learning Rate
    initial_learning_rate = 0.0001
    reduce_Rl=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    # Create the new model
    input2 = tf.keras.Input(shape=(50, 50, 50, 1))
    x = base_model(input2, training = False)
    flat2 = tf.keras.layers.Flatten()(x)
    dense1 = tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(flat2)
    dense2 = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(dense1)
    dense3 = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(dense2)
    dense4 = tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3))(dense3)
    #dense5 = tf.keras.layers.Dense(units=64, activation="relu")(dense4)
    drop2 = tf.keras.layers.Dropout(0.1)(dense4)
    output2 = tf.keras.layers.Dense(units=1)(drop2)

    # Compile the model
    model = tf.keras.Model(input2, output2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='MAE', metrics=['MSE'])

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "3d_mmse_regression_{feature}_{size}.h5", save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    #Fit the data
    history=model.fit(X_train_tot,Y_train_tot, validation_split=0.2, batch_size=32, shuffle=True, epochs=30, callbacks=[checkpoint_cb, early_stopping, reduce_Rl])

    #history contains information about the training
    print(history.history.keys())

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["MSE", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    plt.show()

    #Fine tuning
    base_model.trainable = True
    model.summary()

    checkpoint_tun = tf.keras.callbacks.ModelCheckpoint(
            "3d_mmse_regression_{feature}_{size}_tun.h5", save_best_only=True
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='MAE', metrics=['MSE'])


    history=model.fit(X_train_tot,Y_train_tot, validation_split=0.2, batch_size=32, shuffle=True, epochs=10, callbacks=[checkpoint_tun, early_stopping, reduce_Rl])

    print(history.history.keys())

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["MSE", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    plt.show()

    #Display ROC curve and calculate AUC
    auc = roc_curve(X_test, Y_test, model)

if __name__=='__main__':
    dataset_path_AD_ROI = "AD_CTRL/AD_ROI_TH"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI_TH"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head, dict_age, dict_mmse = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    # import images, labels, file names, age and mmse
    X, _, fnames_AD, fnames_CTRL, file_id, age, mmse = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI,dict_age, dict_mmse , str_1='1', str_2='_')

    X=normalize(X)

    age=np.array(age)
    AGE_MAX=np.max(age)
    age_norm=age/AGE_MAX

    mmse=np.array(mmse)
    MAX_MMSE=np.max(mmse)
    mmse_norm= mmse/MAX_MMSE

    Xctrl = X[144:332,:,:,:]
    age_ctrl = age[144:332]
    AGE_CTRL_MAX=np.max(age_ctrl)
    age_ctrl_norm=age_ctrl/AGE_CTRL_MAX
    '''
    # Train the transfer learning model for the age with AD and CTRL subjects
    training_tl(X, age_norm, 0.15)

    # Train the transfer learning model for the mmse of AD and CTRL subjects
    training_tl(X, mmse_norm, 0.15)
    '''
    # Train the transfer learning model for the age of CTRL subjects
    training_tl(X, age_ctrl_norm, 0.2)
