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
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np
from numpy import interp
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

from data_augmentation import VolumeAugmentation
from input_dati import cut_file_name, read_dataset, import_csv

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
    pred: ???
        the prediction of the CNN
    true: ???
        the label of the image
    Returns
    -------
    dice: float
        Dice index for the array of images
    """
    intersection = 2.0 *np.sum(pred[true==k])
    dice = intersection / (pred.sum() + true.sum())
    return dice

def roc_curve(xtest, ytest, model):
    """
    Display ROC curve and calculate AUC
    Parameters
    ----------
    xtest: 4D np.array
        array containg test images
    ytest: 2D np.array
        array containing test labels
    Returns
    -------
    auc: float
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

def plot_cv_roc(X, y, classifier, n_splits=5, scaler=None):
    """
    plot_cv_roc trains the classifier on X data with y labels, implements the
    k-fold-CV with k=n_splits, may implement a feature scaling function.
    It plots the ROC curves for each k fold and their average and displays
    the corresponding AUC values and the standard deviation over the k folders.
    """
    if scaler:
        model = Pipeline([('scaler', scaler()),
                    ('classifier', classifier)])
    else:
        model = classifier

    try:
        y = y.to_numpy()
        X = X.to_numpy()
    except AttributeError:
        pass

    cv = StratifiedKFold(n_splits)

    tprs = [] #True positive rate
    aucs = [] #Area under the ROC Curve
    interp_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 0
    for train, test in cv.split(X, y):

        y_score = model.predict(X[test])
        fpr, tpr, thresholds = metrics.roc_curve(y[test], y_score)
        #print(f"{fpr} - {tpr} - {thresholds}\n")
        interp_tpr = interp(interp_fpr, fpr, tpr)
        tprs.append(interp_tpr)
        roc_auc = metrics.auc(fpr, tpr)
        #roc_auc = metrics.roc_auc_score(y[test], y_score)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
        i += 1

    plt.legend()
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.show()

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(interp_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(interp_fpr, mean_tpr, color='b',
          label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
          lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC of SVM',fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()


if __name__=='__main__':

    dataset_path_AD_ROI = "AD_CTRL/AD_s3"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_s3"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

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

    auc = roc_curve(X_test, Y_test, model)

    plot_cv_roc(X,Y, model, 5, scaler=None)

    '''
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


    #dice_value=dice_vectorized(Y_train ,model.predict(X_train_dice)>0.1)

    #dice_mean_train = dice_vectorized(Y_train,model.predict(X_train_dice)>0.1).mean()
    #dice_mean_test = dice_vectorized(Y_test,model.predict(X_test_dice)>0.1).mean()

    #print(f'indice di Dice vettorizzato dati di train: {dice_value}')
    #print(f'indice di Dice vettorizzato medio dati di train: {dice_mean_train}')
    #print(f'indice di Dice vettorizzato medio dati di test: {dice_mean_test}')

    X_test = tensorflow.expand_dims(X_test, axis=-1)
    y_pred_test = model.predict(X_test)

    y_pred_bool = y_pred_test > 0.5
    y_test_bool = Y_test > 0.5
    y_corr = np.empty(shape=len(Y_test), dtype=bool)

    for i in range(0,len(y_pred_test)):
        y_corr[i] = (y_test_bool[i] == y_pred_bool[i])

    print(y_corr)

    y_Corr = np.empty(shape=len(Y_test), dtype=int)
    for i in range(0, len(y_corr)):
        if y_corr[i]==True:
            y_Corr[i]=1
        if y_corr[i]==False:
            y_Corr[i]=0

    print(y_Corr)
    '''
