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

try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')


dataset_path_AD_ROI = "AD_CTRL/AD_ROI"
dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI"
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


'''
example_filename = os.path.join(dataset_path_AD_ROI, 'smwc1AD-1_ROI.nii')
img = nib.load(example_filename)
print(img.shape)

img = nib.load(example_filename).get_fdata()
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
ax1.imshow(img[img.shape[0]//2])
ax1.set_title('Image')
ax2.imshow(img[img.shape[0]//2])
ax2.set_title('Mask')
plt.show()
'''
'''
imagesAD_ROI=glob(os.path.join(dataset_path_AD_ROI,'*'))
imagesCTRL_ROI=glob(os.path.join(dataset_path_CTRL_ROI,'smwc1CTRL','*'))

print(f' {len(imagesAD_ROI)}')
print(f' {len(imagesAD_ROI)} matching files found: {imagesAD_ROI[0]}, ..., {imagesAD_ROI[-1]} '  )
'''

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
        Y.append(-1)
    return np.array(X), np.array(Y), fnames_AD, fnames_CTRL

X, Y, fnames_AD, fnames_CTRL = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI)

#Da far diventare un Unittest
print(X.shape, Y.shape)
print(X.min(), X.max(), Y.min(), Y.max())

#print(f' {len(fnames_AD)} matching files found: {fnames_AD[0]}, {fnames_AD[1]}, {fnames_AD[2]}, {fnames_AD[3]} ..., {fnames_AD[50]},...,{fnames_AD[-5]}, {fnames_AD[-4]}, {fnames_AD[-3]}, {fnames_AD[-2]}, {fnames_AD[-1]} '  )
#
#Normalization of intensity voxel values
X=X/X.max()


"""
Divide the dataset in train and test in a static way

"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Da fare diventare un Unittest
print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')

"""
Defining the CNN model
"""

from keras.layers import Conv3D, Input, Dense, MaxPooling3D, BatchNormalization, ReLU
from keras.models import Model, load_model

def make_model(shape=(108, 135, 109, 1)):
    input_tensor = Input(shape=shape)
    hidden = Conv3D(32, (3, 3, 3), strides=1, padding='same', activation='relu')(input_tensor)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    #hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= Flatten()(hidden)
    hidden=  Dense(50, activation='relu')(hidden)
    hidden=  Dense(20, activation='relu')(hidden)
    hidden=  Dense(20, activation='relu')(hidden)
    model = Model(input_tensor, out)

    return model
