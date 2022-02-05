"""
The VolumeAugmentation class shuffles the dataset
and rotates the volume by an angle randomly chosen.
"""
import random
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from scipy import ndimage

#pylint: disable=invalid-name

class VolumeAugmentation():
    """ Compute data augmentation by rotating the volume.

    :Attributes:
        x : np.array
            Array containing 3D images
        y : np.array
            Array containing labels
        shape : tuple
            Image shape

    :Methods:
        shuffle_dataset :
            Shuffle the dataset
        augment :
            Create two arrays cointaing rotated images and labels
        rotate :
            Define some rotation angles and rotate the volume by a few degrees
    """

    def __init__(self, x, y, shape):
        """
        Initialize the class.
        """
        self.x, self.y = x, y
        self.shape = shape

    def shuffle_dataset(self):
        """Shuffle the dataset"""
        self.x, self.y = shuffle(self.x, self.y)

    def augment(self):
        """
        Create two arrays cointaing rotated images and labels.

        :Parameters:
            None

        :Returns:
            X : 3D np.array
                Array containig 3D images
            Y : np.array
                Array containing labels
        """
        X=[]
        Y=[]
        self.shuffle_dataset()
        i = 1
        for image, labels in zip(self.x, self.y):
            X.append(self.rotate(image))
            Y.append(labels)
            print(f'Processing image {i} of {len(self.y)}')
            i+=1
        return np.array(X), np.array(Y)

    def rotate(self, volume):
        """
        Define some rotation angles and rotate the volume by a few degrees.

        :Parameters:
            volume : 3D image
                Image that you want to rotate

        :Returns:
            augmented_volume : 3D image
                Rotated image
        """
        def scipy_rotate(volume):
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
