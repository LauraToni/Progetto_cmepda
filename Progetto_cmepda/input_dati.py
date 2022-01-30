import os
from glob import glob
import numpy as np
import pandas as pd
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

#pylint: disable=invalid-name
#pylint: disable=line-too-long

def cut_file_name (file_name, str_1, str_2):
    """
    Cut the name of the NifTi file to create a string that identify the image easily.

    :Parameters:
        file_name : list of str
            List of str containing file names from csv
        str_1 : str
            Identification string used in the first cut
        str_2 : str
            Identification string used in the second cut
    """
    # First cut
    pos_1 = file_name.index(str_1)
    first_cut = file_name[(pos_1+1):]

    # Second cut
    pos_2 = first_cut.index(str_2)
    second_cut = first_cut[:(pos_2)]
    return second_cut

def read_dataset(dataset_path_AD, dataset_path_CTRL, dic_csv_age, dic_csv_mmse, str_1, str_2, x_id ="AD-", y_id="CTRL-"):
    """
    Load images from NIFTI directory and arrange image names, age and mmse as the csv file.

    :Parameters:
        dataset_path_AD : str
            Directory path for AD images
        dataset_path_CTRL : str
            Directory path for CTRL images
        dic_csv_age : dict
            Dictonary containing the age of AD and CTRL
        dic_sve_mmse : dict
            Dictonary containing the mmse of AD and CTRL
        str_1 : char
            Identification char used to cut the names in the right position.
        str_2 : char
            Identification char used to cut the names in the right position.
        x_id : str
            Identification string in the filename of AD images
        y_id : str
            Identification string in the filename of CTRL images

    :Returns:
        X : np.array
            Array of AD and CTRL images data
        Y: np.array
            Array of labels
        file_names_AD: list of str
            List containig AD images file names
        file_names_CTRL: list of str
            List containig CTRL images file names
        id_list : list of str
            List containing the names of the images arranged as the csv
        file_Age : list of str
            List containing the age of the patients arranged as the csv
        file_Mmse : list of str
            List containing the mmse of the patients arranged as the csv
    """

    file_names_AD = sorted(glob(os.path.join(dataset_path_AD, f"*{x_id}*.nii"  )))
    file_names_CTRL= sorted(glob(os.path.join(dataset_path_CTRL, f"*{y_id}*.nii"  )))

    # Define support lists
    X_list = []
    Y_list = []
    id_list = []
    file_Age = []
    file_Mmse = []
    i=1

    for fname_AD in file_names_AD:
        X_list.append(nib.load(fname_AD).get_fdata())
        Y_list.append(1)

        # Cut AD filenames
        name = cut_file_name(fname_AD, str_1, str_2)

        # Fill lists
        id_list.append(name)
        file_Age.append(dic_csv_age[name])
        file_Mmse.append(dic_csv_mmse[name])

        # Load AD images
        print(f'Caricamento immagine {name} ({i} di 144)')
        i+=1
    i=1
    for fname_CTRL in file_names_CTRL:
        X_list.append(nib.load(fname_CTRL).get_fdata())
        Y_list.append(0)

        # Cut CTRL filenames
        name = cut_file_name(fname_CTRL, str_1, str_2)

        # Fill lists
        id_list.append(name)
        file_Age.append(dic_csv_age[name])
        file_Mmse.append(dic_csv_mmse[name])

        # Load CTRL images
        print(f'Caricamento immagine {name} ({i} di 189)')
        i+=1
    return np.array(X_list), np.array(Y_list), file_names_AD, file_names_CTRL, id_list, file_Age, file_Mmse

def import_csv(path):
    """
    Import metadata from csv file.

    :Parameters:
        path: str
            Directory path of the metadata file

    :Returns:
        df : pandas DataFrame
            Data structure with labeled axis
        head : pandas Header
            List of strings used as column names
        dic_age : dict
            Dictonary containing the age of the patients
        dic_mmse : dict
            Dictonary containing the mmse of the patients
    """

    # Read the csv
    df = pd.read_csv(path, sep=',')
    head=df.head()

    # Create the arrays containing names, age and mmse
    file_id_list = df['ID'].tolist()
    file_id_csv = np.array(file_id_list)
    file_age_list = df['AGE'].tolist()
    file_age_csv = np.array(file_age_list)
    file_mmse_list = df['MMSE'].tolist()
    file_mmse_csv = np.array(file_mmse_list)

    # Use a dictonary to create a corrispondence between names and age or mmse
    dic_age = {}
    dic_mmse = {}

    # Fill the dictonary
    for i in range(0, len(file_id_csv)):
        dic_age[file_id_csv[i]] = file_age_csv[i]
        dic_mmse[file_id_csv[i]] = file_mmse_csv[i]

    return df, head, dic_age, dic_mmse

if __name__=='__main__':

    # Define the dataset path
    dataset_path_AD_ROI = "AD_CTRL/AD_ROI_TH"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI_TH"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head, dict_age, dict_mmse = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])


    # import images, labels, file names, age and mmse
    X, Y, fnames_AD, fnames_CTRL, file_id, file_age, file_mmse = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI, dict_age, dict_mmse , str_1='1', str_2='_')
