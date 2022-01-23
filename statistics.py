import os
import PIL
import zipfile
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
from sklearn import metrics
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

#from data_augmentation import VolumeAugmentation
#from input_dati import cut_file_name, read_dataset, import_csv
#from CNN import normalize

def dice(pred, true, k = 1):
    """
    Calculate Dice index for a single image

    Parameters
    ----------
    pred: float
        the prediction of the CNN
    true: int
        the label of the image
    Returns
    -------
    dice: float
        Dice index for the image
    """
    intersection = np.sum(pred[true==k]) * 2.0
    dice_coef = intersection / (np.sum(pred) + np.sum(true))
    return dice_coef

def dice_vectorized(pred, true, k = 1):
    """
    Calculate Dice index for an array of images

    Parameters
    ----------
    pred : ???
        the prediction of the CNN
    true : ???
        the label of the image
    Returns
    -------
    dice: float
        Dice index for the array of images
    """
    intersection = 2.0 *np.sum(pred[true==k])
    dice = intersection / (pred.sum() + true.sum())
    return dice

def roc_curve(xtest, ytest):
    """
    Display ROC curve and calculate AUC

    Parameters
    ----------
    xtest : 4D np.array
        array containg test images
    ytest : 2D np.array
        array containing test labels
    Returns
    -------
    auc : float
        area under the ROC curve
    """
    y_score = model.predict(xtest)
    fpr, tpr, thresholds = metrics.roc_curve(ytest, y_score)

    auc = metrics.roc_auc_score(ytest, y_score)
    print(f'AUC: {auc}')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc,)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.show()

    return auc


if __name__=='__main__':

    dataset_path_AD_ROI = "AD_CTRL/AD_s3"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_s3"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"
    '''
    # Import csv data
    df, head, dic_info = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    # import images, labels and file names
    X_o, Y, fnames_AD, fnames_CTRL, file_id, file_age = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI, dic_info, str_1='1', str_2='.')

    X_o=normalize(X_o)

    X=X_o[:,36:86,56:106,24:74] #ippocampo

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=11)
    print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
    print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')


    model = tensorflow.keras.models.load_model("3d_image_classification.h5")
    model.summary()

    auc = roc_curve(X_test, Y_test)
    #Calcolo indice di Dice
    idx=67
    xtrain = X_train[idx][np.newaxis,...]
    ytrain = Y_train[idx][np.newaxis,...]
    print(Y_train[idx].shape, ytrain.shape)

    ypred = model.predict(xtrain).squeeze()>0.1
    ytrue = Y_train[idx].squeeze()

    dice_value = dice(ypred, ytrue)
    print(f'Indice di DICE:{dice_value}')

    X_train_dice = tensorflow.expand_dims(X_train, axis=-1)
    X_test_dice = tensorflow.expand_dims(X_test, axis=-1)


    dice_value=dice_vectorized(Y_train ,model.predict(X_train_dice)>0.1)

    dice_mean_train = dice_vectorized(Y_train,model.predict(X_train_dice)>0.1).mean()
    dice_mean_test = dice_vectorized(Y_test,model.predict(X_test_dice)>0.1).mean()

    print(f'indice di Dice vettorizzato dati di train: {dice_value}')
    print(f'indice di Dice vettorizzato medio dati di train: {dice_mean_train}')
    print(f'indice di Dice vettorizzato medio dati di test: {dice_mean_test}')
    '''
