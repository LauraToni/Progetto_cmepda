import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


dataset_path_AD_ROI = "AD_CTRL/AD_ROI"
dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI"

try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

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
imagesAD_ROI=glob(os.path.join(dataset_path_AD_ROI,'*'))
imagesCTRL_ROI=glob(os.path.join(dataset_path_CTRL_ROI,'smwc1CTRL','*'))

print(f' {len(imagesAD_ROI)}')
print(f' {len(imagesAD_ROI)} matching files found: {imagesAD_ROI[0]}, ..., {imagesAD_ROI[-1]} '  )


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
        array of AD images data
    Y: np.array
        array of CTRL images data

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
    for fname_CTRL in fnames_CTRL:
        Y.append(nib.load(fname_CTRL).get_fdata())
    return np.array(X), np.array(Y), fnames_AD, fnames_CTRL

X,Y, fnames_AD, fnames_CTRL = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI)

print(f' {len(fnames_AD)} matching files found: {fnames_AD[0]}, ..., {fnames_AD[50]},..., {fnames_AD[-1]} '  )
print(f' {len(fnames_CTRL)} matching files found: {fnames_CTRL[0]}, ..., {fnames_CTRL[50]},..., {fnames_CTRL[-1]} '  )
