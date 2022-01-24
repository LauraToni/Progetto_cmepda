""" Class VolumeAugmentation shuffles the dataset and rotates the volume by an angle chosen randomly"""
import random
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from scipy import ndimage

try:
    import zipfile
except:
    raise ImportError('Install zipfile')

#pylint: disable=invalid-name

class VolumeAugmentation():
    """ Data augmentation class for a CNN """

    def __init__(self, x, y, shape):
        """
        Initialize the sequence
        Parameters:
        x : np.array
            array containing 3D images
        y : np.array
            array containing labels
        shape : tuple
            image shape

        """
        self.x, self.y = x, y
        self.shape = shape

    def shuffle_dataset(self):
        """Shuffle the dataset"""
        self.x, self.y = shuffle(self.x, self.y)

    def augment(self):
        """
        Create two arrays cointaing rotated images and labels
        Parameters
        ----------
        None

        Returns
        ----------
        X : 3D np.array
            array containig 3D images
        Y : np.array
            array containing labels

        """
        X=[]
        Y=[]
        self.shuffle_dataset()
        for image, labels in zip(self.x, self.y):
            X.append(self.rotate(image))
            Y.append(labels)
            print("sto processando l'immagine")
        return np.array(X), np.array(Y)

    def rotate(self, volume):
        """
        Rotate the volume by a few degrees
        Parameters
        ----------
        volume : 3D image
            image that you want to rotate

        Returns
        ----------
        augmented_volume : 3D image
            rotated image

        """
        def scipy_rotate(volume):
            """
            Define some rotation angles and rotate the volume
            Parameters
            ----------
            volume : 3D image
                image that you want to rotate

            Returns
            ----------
            volume : 3D image
                rotated image

            """
            # define some rotation angles
            angles = [-20, -10, -5, 5, 10, 20]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            return volume
        augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
        return augmented_volume
