import os
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn import metrics
import string
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')
from nibabel.testing import data_path

#from data_augmentation import VolumeAugmentation

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#pylint: disable=invalid-name
#pylint: disable=line-too-long

def cut_file_name (file_name):
    str_1 = '1'
    pos_1 = file_name.index(str_1)
    first_cut = file_name[(pos_1+1):]

    str_2 = '_'
    pos_2 = first_cut.index(str_2)
    second_cut = first_cut[:(pos_2)]
    return second_cut

def cut_file_csv (file_name):

    first_cut = file_name[(0):]
    second_cut = first_cut[:(-1)]
    return second_cut


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

    file_names_AD: list (?)
        list containig AD images file names
    file_names_CTRL: list (?)
        list containig CTRL images file names

    """

    file_names_AD = sorted(glob(os.path.join(dataset_path_AD, f"*{x_id}*.nii"  )))
    file_names_CTRL= sorted(glob(os.path.join(dataset_path_CTRL, f"*{y_id}*.nii"  )))
    X = []
    Y = []
    id = []

    for fname_AD in file_names_AD:
        X.append(nib.load(fname_AD).get_fdata())
        Y.append(1)
        id.append(cut_file_name(fname_AD))
    for fname_CTRL in file_names_CTRL:
        X.append(nib.load(fname_CTRL).get_fdata())
        Y.append(0)
        id.append(cut_file_name(fname_CTRL))
    return np.array(X), np.array(Y), file_names_AD, file_names_CTRL, id

def import_csv(path):
    """
    Import metadata from csv file
    Parameters
    ----------
    path: str
        directory path of the metadata file
    Returns
    -------
    df :

    head :

    """
    df = pd.read_csv(path, sep=',')
    head=df.head()
    return df, head

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
