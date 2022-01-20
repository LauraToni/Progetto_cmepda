""" Convolutional neural network applied to neurological AD and CTRL images"""
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
from input_dati.py import read_dataset,import_csv, cut_file_csv, cut_file_name

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#pylint: disable=invalid-name
#pylint: disable=line-too-long

def normalize(x):
    """
    Normalize the intensity of every pixel in the image
    Parameters
    ----------
    x :
    Returns
    -------
     :

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
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.layers.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer='l2')(x)
    #x=Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model_cnn = tf.keras.models.Model(inputs, outputs, name="3dcnn")
    return model_cnn

def get_model_art(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.layers.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=2)(x)

    '''
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2,  strides=1)(x)
    '''

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model_art_cnn = tf.keras.models.Model(inputs, outputs, name="3dcnn")
    return model_art_cnn

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice_coef = intersection / (np.sum(pred) + np.sum(true))
    return dice_coef

if __name__=='__main__':
    dataset_path_AD_ROI = "AD_CTRL/AD_ROI"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    file_id_list = df['ID'].tolist()
    file_id_csv = np.array(file_id_list)
    file_age_list = df['AGE'].tolist()
    file_age_csv = np.array(file_age_list)
    file_mmse_list = df['MMSE'].tolist()
    file_mmse_csv = np.array(file_mmse_list)

    # import images, labels and file names
    X_o, Y, fnames_AD, fnames_CTRL, file_id = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI)
    #Normalization of intensity voxel values
    X_o=normalize(X_o)

    # Define ROI
    #X=X_o[:,36:86,56:106,24:74] #ippocampo
    #X=X_o[:,11:109,12:138,24:110] #bordi neri precisi
    X=X_o[:,20:100,20:130,20:100]
    #X=X_o

    # Divide the dataset in train, validation and test in a static way
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    #Augment the data using VolumeAugmentation class
    mass_gen = VolumeAugmentation(X_train, Y_train, shape=(X.shape[1], X.shape[2], X.shape[3]))
    array_img, labels = mass_gen.augment()

    # Create an array containing both original and augmented data
    X_train_tot, Y_train_tot=stack_train_augmentation(X_train, array_img, Y_train, labels)

    # Build the model
    model = get_model_art(width=X.shape[1], height=X.shape[2], depth=X.shape[3])
    model.summary()

    # Set the learning Rate
    initial_learning_rate = 0.001
    ReduceLROnPlateau=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['MAE'])

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "model.{epoch:02d}-{val_MAE:.4f}_C8_C8_C16_C16_C32_C32_D32_Hipp_art.h5", save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    #Fit the data
    #The sample is automatically split in two so that 50% of it is used for validation and the other half for training
    history=model.fit(X_train_tot,Y_train_tot, validation_split=0.1, batch_size=32, shuffle=TRUE, epochs=20, callbacks=[early_stopping, ReduceLROnPlateau])

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

    #history = model.fit(X_train_tot,Y_train_tot, validation_split=0.1, batch_size=32, epochs=10, callbacks=[checkpoint_cb, ReduceLROnPlateau])

    # Create reshaped arrays to compute DICE coefficient
    idx=67
    xtrain = X_train_tot[idx][np.newaxis,...]
    ytrain = Y_train_tot[idx][np.newaxis,...]
    print(Y_train_tot[idx].shape, ytrain.shape)
    ypred = model.predict(xtrain).squeeze()>0.1
    ytrue = Y_train_tot[idx].squeeze()

    # Compute DICE coefficient
    dice_value = dice(ypred, ytrue)
    print(f'Indice di DICE:{dice_value}')
    print(ypred.shape, ytrue.shape)

    # Use the model to predict the probability that a given y value is 1
    y_score = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_score)

    #calculate AUC of model
    auc = metrics.roc_auc_score(Y_test, y_score)

    #print AUC score
    print(auc)

    #plot ROC_curve
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
