import unittest
import numpy as np
from CNN import normalize
from input_dati import read_dataset, import_csv
import string
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from data_augmentation import VolumeAugmentation

class TestCNN(unittest.TestCase):
    """
    Class for testing the functionalities of our code
    """
    def setUp(self):
        """
        Set up the test
        """
        self.NAD=5
        self.NCTRL=5
        self.dataset_path_AD_ROI = "AD_CTRL/AD_ROIL"
        self.dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROIL"
        self.dataset_path_metadata = "AD_CTRL_metadata_labels.csv"
        _, _, self.dica , self.dicm = import_csv(self.dataset_path_metadata)
        self.x, self.y, self.fnames_AD, self.fnames_CTRL, self.file_id, self.age, self.mmse =read_dataset(self.dataset_path_AD_ROI, self.dataset_path_CTRL_ROI, self.dica, self.dicm, str_1='1', str_2='_')
        self.volume = VolumeAugmentation(self.x, self.y, shape=(self.x))

    def test_shape(self):
        self.assertEqual( (self.x.shape[0]), len(self.y) )
        self.assertEqual( (len(self.x.shape)), 4)
        self.assertEqual( (len(self.y.shape)), 1)

    def test_len(self):
        self.assertEqual( len(self.fnames_AD), self.NAD)
        self.assertEqual( len(self.fnames_CTRL), self.NCTRL)
        self.assertEqual( len(self.file_id), len(self.age))
        self.assertEqual( len(self.file_id), len(self.mmse))

    def test_metadata(self):
        df, head,_ ,_ = import_csv(self.dataset_path_metadata)
        #count the entries grouped by the diagnostic group
        print(df.groupby('DXGROUP')['ID'].count())

    def test_max_min (self):
        self.assertEqual( (self.y.max()) , 1)
        self.assertEqual( (self.y.min()) , 0)
        nx=normalize(self.x)
        self.assertEqual( (nx.max()) , 1)
        self.assertEqual( (nx.min()) , 0)

    def test_augmentation(self):
        self.assertEqual( len(self.volume.augment()[1]), len(self.y))
        self.assertEqual( self.volume.augment()[0].shape, self.x.shape)

if __name__=='__main__':
   unittest.main()
