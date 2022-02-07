""" Statistics analysis """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from input_dati import read_dataset, import_csv
SOGLIA = 0.4273
#pylint: disable=invalid-name
#pylint: disable=line-too-long

def normalize(x):
    """
    Normalize the intensity of every pixel in the image.

    :Parameters:
        x : 4D np.array
            array containing the images
    :Returns:
        x : 4D np.array
            array containg the normalized images

    """
    return x/x.max()

def dice(pred, true, k = 1):
    """
    Calculate Dice index for a single image.

    :Parameters:
        pred: float
            the prediction of the CNN
        true: int
            the label of the image
    :Returns:
        dice: float
            Dice index for the image
    """
    intersection = np.sum(pred[true==k]) * 2.0
    # Compute dice coefficient
    dice_coef = intersection / (np.sum(pred) + np.sum(true))
    return dice_coef

def roc_curve(xtest, ytest, model):
    """
    Display ROC curve and calculate AUC.

    :Parameters:
        xtest: 4D np.array
            array containg test images
        ytest: 2D np.array
            array containing test labels
    :Returns:
        auc: float
            area under the ROC curve
    """
    # Compute y scores
    y_score = model.predict(xtest)

    # Compute roc curve
    fpr, tpr, threasholds = metrics.roc_curve(ytest, y_score)
    print(f"{fpr} - {tpr} - {threasholds}\n")
    # Compute area under the curve
    auc = metrics.roc_auc_score(ytest, y_score)
    print(f'AUC: {auc}')

    # Choose the threashold
    j=0
    k=0
    while (k==0):
        if ((tpr[j]> 0.80) and (tpr[j] < 0.85)):
            indice=j
            k=k+1
        j=j+1

    print(f'La soglia è: {threasholds[indice]}')
    print(f'FPR: {fpr[indice]}')
    print(f'sensitività: {tpr[indice]}')
    print(f'specificità: {1 - fpr[indice]}')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc,)
    #plt.plot(fpr, threasholds, color='c', label='threashold')
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.axvline(fpr[indice], linestyle='--', color='green')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.show()

    return auc

def permutation_age(df, Nperm):
    '''
    Compute the permutation test on the difference between predicted and real age.
    The null hypothesis is that there's no difference between the distributions of AD and CTRL,
    against the hipothesis that there's a difference for images belonging to one category.

    :Parameters:
        df : pandas dataframe
            Dataframe containing the features of all the test images
        Nperm : int
            Number of permutations
    :Returns:
        p_value : float
            p_value of the null hypothesis
    '''
    feature_AD_pred = df['Age_AD_pred']
    feature_CTRL_pred = df['Age_CTRL_pred']
    feature_AD_test = df['Age_AD_test']
    feature_CTRL_test = df['Age_CTRL_test']

    # AD mean age difference
    feature_AD_diff = feature_AD_pred-feature_AD_test
    feature_AD_mean = feature_AD_diff.mean()

    # CTRL mean age difference
    feature_CTRL_diff = feature_CTRL_pred-feature_CTRL_test
    feature_CTRL_mean = feature_CTRL_diff.mean()

    feature_diff = feature_CTRL_mean - feature_AD_mean
    n_perm = Nperm

    feature_all = np.append(feature_AD_diff, feature_CTRL_diff)

    feature_diff_perm = []
    for i in range(n_perm):
        perm_i = np.random.permutation(feature_all)
        avg_A = perm_i[1:feature_AD_diff.shape[0]].mean()
        avg_B = perm_i[feature_AD_diff.shape[0]:len(feature_all)].mean()
        #avg_B = perm_i[feature_AD_diff.shape[0]:n_examples].mean()
        feature_diff_perm = np.append(feature_diff_perm, avg_A - avg_B)
    feature_diff_perm.shape

    plt.title('Permutation test age')
    _ = plt.hist(feature_diff_perm, 25, histtype='step')
    plt.xlabel(f'Ages [yrs] ')
    plt.ylabel('Occurrences')
    plt.axvline(feature_diff, linestyle='--', color='red')
    plt.show()

    feature_diff_perm[abs(feature_diff_perm) > abs(feature_diff)].shape[0]

    r = feature_diff_perm[feature_diff_perm > feature_diff].shape[0]
    p_value = (r + 1 )/ (n_perm +1)
    if r == 0:
        print(f'The p value is p < {p_value:.3f}')
    else:
        print(f'The p value is p = {p_value:.3f}')
    if p_value < 0.05:
        print('The difference between the mean weight loss of the two groups is statistically significant! ')
    else:
        print('The null hypothesis cannot be rejected')

    return p_value

def dataframe_test(x, y, age, mmse):
    """
    Create different dataframes containig labels, age and MMSE and Confronto_predizione.
    Confronto_predizione is an array that is 1 if the prediction
    of the model is correct and 0 if it is wrong.

    :Parameters:
        x : 4D array
            Array containg the images
        y : 1D array
            Array containg the labels of the images
        age : Tupla
            Tupla containing the age relative to the images
        mmse : Tupla
            Tupla containg the MMSE relative to the images
        model : Keras model
            base model containing CNN layers
        agemodel : Keras model
            model used to predict age
        mmsemodel : Keras model
            model used to predict mmse
    :Returns:
        dataFrame_total : pandas dataframe
            Dataframe contining the feautures of all test's images
        dataFrame_ctrl : pandas dataframe
            Dataframe contining the feautures of the CTRL and AD test's images
        dataFrame_ad_total : pandas dataframe
            Dataframe contining the feautures of all AD images
        dataFrame_AD : pandas dataframe
            Dataframe contaning the features of test images belongig to AD category
        dataFrame_CTRL : pandas dataframe
            Dataframe contaning the feature of test images belongig to CTRL category
        dataFrame_diff : pandas dataframe
            Dataframe containing age difference between predicted and real AD and CTRL
    """
    #Divide in train and test
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.15, random_state=14)

    _, age_test_total, _, mmse_test = train_test_split(age, mmse, test_size=0.15, random_state=14)

    x_ctrl_train, x_ctrl_test, age_ctrl_train, age_ctrl_test = train_test_split(x[144:333,:,:,:], age[144:333], test_size=0.20, random_state=14)

    age_ad_test = age[0:38]
    x_ad_test = x[0:38,:,:,:]

    # Expand X test dimension to compute y, age and mmse prediction
    X_test = tf.expand_dims(x_test, axis=-1)
    X_ctrl_test = tf.expand_dims(x_ctrl_test, axis=-1)
    X_ad_test = tf.expand_dims(x_ad_test, axis=-1)
    X_ad_total_test = tf.expand_dims(x[0:144,:,:,:], axis=-1)

    # prediction label CTRL and AD
    y_pred_test = model.predict(X_test)
    # prediction of the age of all the examples
    age_pred = age_model.predict(X_test)
    # prediction of the mmse of all the examples
    mmse_pred = mmse_model.predict(X_test)
    # prediction of the age for ctrl examples
    age_ctrl_pred = age_model_ctrl.predict(X_ctrl_test)
    # prediction of the age for ad examples
    age_ad_pred = age_model_ctrl.predict(X_ad_test)
    #prediction of the age for all the AD examples
    age_ad_total_pred = age_model_ctrl.predict(X_ad_total_test)

    # Squeeze dimensions
    y_pred_test = np.squeeze(y_pred_test)
    age_pred = np.squeeze(age_pred)
    mmse_pred = np.squeeze(mmse_pred)
    age_ctrl_pred = np.squeeze(age_ctrl_pred)
    age_ad_pred = np.squeeze(age_ad_pred)
    age_ad_total_pred = np.squeeze(age_ad_total_pred)

    mmse = np.array(mmse)
    age = np.array(age)

    # Return to original scale
    age_pred = age_pred*100
    mmse_pred = mmse_pred*30
    age_ctrl_pred = age_ctrl_pred*100
    age_ad_pred = age_ad_pred*100
    age_ad_total_pred = age_ad_total_pred*100

    # Difference between predicted and real AD age
    diff_age_AD = age_ad_pred - age_ad_test
    # Difference between predicted and real CTRL age with ctrl dataset
    diff_age_CTRL = age_ctrl_pred - age_ctrl_test
    # Difference age
    diff_age = np.append(diff_age_AD, diff_age_CTRL)
    diff_label = []

    for i in range(0, len(diff_age)):
        if i<=len(diff_age_AD):
            diff_label.append('AD')
        else:
            diff_label.append('CTRL')

    y_conf = np.empty(shape=len(y_test), dtype=bool)

    for i in range(0,len(y_pred_test)):
        y_conf[i] = (y_test[i] == (y_pred_test[i]>SOGLIA))

    y_Conf = np.empty(shape=len(y_test), dtype=int)
    for i in range(0, len(y_conf)):
        if y_conf[i] == True:
            y_Conf[i]=1
        if y_conf[i] == False:
            y_Conf[i]=0

    # Create dataframe labels
    d_total = {'labels_test': y_test, 'y_pred': y_pred_test ,'Confronto_predizione': y_Conf, 'Age_test' : age_test_total, 'Age_pred' : age_pred, 'MMSE_test' : mmse_test, 'MMSE_pred' : mmse_pred }
    dataFrame_total = pd.DataFrame(data=d_total)

    d_ctrl = {'Age_CTRL_test': age_ctrl_test, 'Age_CTRL_pred': age_ctrl_pred, 'Age_AD_test': age_ad_test, 'Age_AD_pred': age_ad_pred}
    dataFrame_ctrl=pd.DataFrame(data=d_ctrl)

    d_ad_totali = {'Age_AD_total_test': age[0:144], 'Age_AD_total_pred': age_ad_total_pred}
    dataFrame_ad_total=pd.DataFrame(data=d_ad_totali)

    d_diff = {'DX_GROUP': diff_label, 'Diff_Age': diff_age}
    dataFrame_diff=pd.DataFrame(data=d_diff)

    dataFrame_AD = dataFrame_total[dataFrame_total.labels_test == 1]
    dataFrame_CTRL = dataFrame_total[dataFrame_total.labels_test == 0]

    # Difference between predicted and real AD age
    diff_age_AD_model = dataFrame_AD.Age_pred - dataFrame_AD.Age_test
    # Difference between predicted and real CTRL age with ctrl dataset
    diff_age_CTRL_model = dataFrame_CTRL.Age_pred - dataFrame_CTRL.Age_test
    # Difference age
    diff_age_model = np.append(diff_age_AD_model, diff_age_CTRL_model)
    diff_label_model = []

    for i in range(0, len(diff_age_model)):
        if i<=len(diff_age_AD_model):
            diff_label_model.append('AD')
        else:
            diff_label_model.append('CTRL')

    d_diff_model = {'DX_GROUP': diff_label_model, 'Diff_Age': diff_age_model}
    dataFrame_diff_model=pd.DataFrame(data=d_diff_model)

    return dataFrame_total, dataFrame_ctrl, dataFrame_ad_total, dataFrame_AD, dataFrame_CTRL, dataFrame_diff, dataFrame_diff_model

def retta(x, m, q):
    """
    Defines a linear function.

    :Parameters:
        x : 1D np.array
            array containing x data
        m : float
            angolar coefficient of the line
        q : float
            intercept of the line

    :Returns:
        mx + q : linear function

    """
    return m*x + q

def fit(xfit, yfit):
    """
    Fit the data.

    :Parameters:
        xfit : 1D np.array
            Array containing x values to fit
        yfit : 1D np.array
            Array containing y values to fit

    :Returns:
        popt : np.array
            Array containinf the oprimal parameters
    """
    #Fit
    popt, pcov = curve_fit(retta, xfit, yfit, p0=[1., 0.])
    m, q = popt
    sigma_m, sigma_q = np.sqrt(pcov.diagonal())
    print(f'm = {m:.2f} +- {sigma_m:.2f} ')
    print(f'q = {q:.2f} +- {sigma_q:.2f} [yrs]' )
    return popt

def permutation(df, Nperm, feature='Age_test'):
    '''
    Compute the permutation test with test image features.
    The null hypothesis is that the mean feature distribution of the correcly predicted images
    the wrongly predicted ones are the same, against the hypothesis that they are different.
    The p-value for the null hypothesis is returned.
    :Parameters:
        df : pandas dataframe
            Dataframe containing the features of all the test images
        Nperm : int
            Number of permutations
        feature : string
            Feature labels
    :Returns:
        p_value : float
            p_value of the null hypothesis
    '''
    feature_pred = df[df['Confronto_predizione'] == 1][feature]
    feature_no_pred = df[df['Confronto_predizione'] == 0][feature]

    #media dell'età delle persone predette correttamente
    feature_pred_mean=feature_pred.mean()

    #media dell'età delle persone NON predette correttamente
    feature_no_pred_mean=feature_no_pred.mean()

    #Differenza delle età medie dei gruppo predetti correttamente e non correttamente
    feature_diff=np.absolute(feature_pred_mean - feature_no_pred_mean)
    n_perm = Nperm
    n_examples=df.shape[0]

    feature_all = np.append(feature_pred, feature_no_pred)

    feature_diff_perm = []
    for i in range(n_perm):
        perm_i = np.random.permutation(feature_all)
        avg_A = perm_i[1:feature_pred.shape[0]].mean()
        avg_B = perm_i[feature_pred.shape[0]:n_examples].mean()
        feature_diff_perm = np.append(feature_diff_perm, avg_A - avg_B)
    feature_diff_perm.shape

    plt.title('Permutation test')
    _ = plt.hist(feature_diff_perm, 25, histtype='step')
    plt.xlabel(f'Difference of {feature} means')
    plt.ylabel('Occurrences')
    plt.axvline(feature_diff, linestyle='--', color='red')
    plt.show()

    feature_diff_perm[abs(feature_diff_perm) > abs(feature_diff)].shape[0]

    r = feature_diff_perm[feature_diff_perm > feature_diff].shape[0]
    p_value = (r + 1 )/ (n_perm +1)
    if r == 0:
        print(f'The p value is p < {p_value:.3f}')
    else:
        print(f'The p value is p = {p_value:.3f}')
    if p_value < 0.025:
        print('The difference between the mean weight loss of the two groups is statistically significant! ')
    else:
        print('The null hypothesis cannot be rejected')

    return p_value

if __name__=='__main__':

    dataset_path_AD_ROI = "AD_CTRL/AD_ROI_TH"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI_TH"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head, dict_age, dict_mmse = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    # import images, labels, file names, age and mmse
    X, Y, fnames_AD, fnames_CTRL, file_id, age, mmse = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI,dict_age, dict_mmse , str_1='1', str_2='_')
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=14)

    #Loading the models
    model = tf.keras.models.load_model("Modelli/3d_CNN_0.327_Hipp_finale.h5")
    #model = tf.keras.models.load_model("Modelli/3d_CNN_0.48_VOID_15.h5")
    model.summary()

    age_model = tf.keras.models.load_model("Modelli/3d_regression_Age_15_0.0052_finale.h5")
    age_model.summary()

    mmse_model = tf.keras.models.load_model("Modelli/3d_regression_MMSE_15_0.017_finale.h5")
    mmse_model.summary()

    age_model_ctrl = tf.keras.models.load_model("Modelli/3d_regression_AgeCTRL_20_0.0086_finale.h5")
    #age_model_ctrl = tf.keras.models.load_model("Modelli/3d_regression_AGECTRL_20_0.0083_VOID_tun_finalissima.h5")
    age_model_ctrl.summary()

    auc = roc_curve(X_test, Y_test, model)

    #Calcolo indice di Dice

    xtrain = np.expand_dims(X_train, axis=-1)
    ytrain = np.expand_dims(Y_train, axis=-1)
    #print(Y_train[idx].shape, ytrain.shape)

    yprob = model.predict(xtrain).squeeze()
    ypred = model.predict(xtrain).squeeze()>SOGLIA
    ytrue = Y_train.squeeze()

    # Compute dice coeffcient
    dice_value = dice(ypred, ytrue)
    print(f'Indice di DICE:{dice_value}')

    #Accuracy
    accuracy = metrics.accuracy_score(ytrue, ypred)
    print(f'Accuracy:{accuracy}')


    df_total, df_CTRL, df_AD_total, df_AD, df_ctrl, df_diff_age, df_diff_age_model=dataframe_test(X, Y, age, mmse)
    print(df_total.head())

    #Correlation dataset total
    #Correlation between the output of the classifier, age and mmse
    print(df_total.drop(columns=['labels_test', 'Confronto_predizione', 'Age_pred', 'MMSE_pred'], axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe total classication')
    sns_heatmap=sns.heatmap(df_total.drop(columns=['labels_test', 'Confronto_predizione', 'Age_pred', 'MMSE_pred'], axis=1).corr(), vmin=-1, vmax=1, annot=True, )
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #Correlation between the output of the predicted age and the real age and mmse
    print(df_total.drop(columns=['labels_test', 'y_pred', 'Confronto_predizione', 'MMSE_pred'], axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe total age regression')
    sns_heatmap=sns.heatmap(df_total.drop(columns=['labels_test', 'y_pred', 'Confronto_predizione', 'MMSE_pred'], axis=1).corr(), vmin=-1, vmax=1, annot=True)
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    # Correlation between the output of the predicted mmse and the real age and mmse
    print(df_total.drop(columns=['labels_test', 'y_pred','Confronto_predizione', 'Age_pred'], axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe total mmse regression')
    sns_heatmap=sns.heatmap(df_total.drop(columns=['labels_test', 'y_pred', 'Confronto_predizione', 'Age_pred'], axis=1).corr(), vmin=-1, vmax=1, annot=True)
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #Correlation CTRL dataset
    #Correlation between the output of the predicted age and the real age for ctrl
    print(df_CTRL.drop(columns=['Age_AD_test', 'Age_AD_pred'], axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe age regression ctrl')
    sns_heatmap=sns.heatmap(df_CTRL.drop(columns=['Age_AD_test', 'Age_AD_pred'], axis=1).corr(), vmin=-1, vmax=1,annot=True)
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #Correlation between the output of the predicted age and the real age for ad
    print(df_CTRL.drop(columns=['Age_CTRL_test', 'Age_CTRL_pred'], axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe age regression ctrl')
    sns_heatmap=sns.heatmap(df_CTRL.drop(columns=['Age_CTRL_test', 'Age_CTRL_pred'], axis=1).corr(), vmin=-1, vmax=1,annot=True)
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #Correlation AD total dataset
    #Correlation between the output of the predicted age and the real age for ad total
    print(df_AD_total.corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe age regression ctrl')
    sns_heatmap=sns.heatmap(df_AD_total.corr(), vmin=-1, vmax=1, annot=True)
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()


    color = df_total.labels_test.apply(lambda x:'blue' if x == 0 else 'red')

    #Scatter Plot

    #Scatter plot dataframe total age
    ax = df_total.plot(x='Age_test', y='Age_pred', kind='scatter', color=color)
    ax.grid()

    #Buil the legend: blu --> CTRL, red --> AD
    red_patch = mpatches.Patch(color='red', label='AD')
    blue_patch = mpatches.Patch(color='blue', label='CTRL')
    patches = [red_patch, blue_patch]
    legend = ax.legend(handles=patches,loc='upper left')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Scatter plot age total')
    plt.show()

    #Scatter plot mmse
    ax = df_total.plot(x='MMSE_test', y='MMSE_pred', kind='scatter', color=color, label=True)
    ax.grid()

    #Buil the legend: blu --> CTRL, red --> AD
    legend = ax.legend(handles=patches,loc='upper left')
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.title('Scatter plot mmse')
    plt.show()

    #Scatter plot dataframe ctrl age for ctrl
    ax = df_CTRL.plot(x='Age_CTRL_test', y='Age_CTRL_pred', kind='scatter', color='blue')
    ax.grid()
    legend = ax.legend(loc='upper left')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Scatter plot age CTRL')
    popt_ctrl=fit(np.array(df_CTRL['Age_CTRL_test']), np.array(df_CTRL['Age_CTRL_pred']))
    lin=np.linspace(0,100,1000)
    plt.plot(lin, retta(lin, *popt_ctrl), color='green', label='linear fit')
    plt.show()


    #Scatter plot dataframe ctrl age for ad
    ax = df_CTRL.plot(x='Age_AD_test', y='Age_AD_pred', kind='scatter', color='red')
    ax.grid()
    legend = ax.legend(loc='upper left')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Scatter plot age AD')
    popt_ad=fit(np.array(df_CTRL['Age_AD_test']),np.array(df_CTRL['Age_AD_pred']))
    lin=np.linspace(0,100,1000)
    plt.plot(lin, retta(lin, *popt_ad), color='green', label='linear fit')
    plt.show()

    #Scatter plot dataframe AD total age for ad
    ax = df_AD_total.plot(x='Age_AD_total_test', y='Age_AD_total_pred', kind='scatter', color='red')
    ax.grid()
    legend = ax.legend(loc='upper left')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Scatter plot age AD total')
    popt_ad_total=fit(np.array(df_AD_total['Age_AD_total_test']),np.array(df_AD_total['Age_AD_total_pred']))
    lin=np.linspace(0,100,1000)
    plt.plot(lin, retta(lin, *popt_ad_total), color='green', label='linear fit')
    plt.show()

    '''
    Compute MAE e RMSE for MMSE and Age total and Age CTRL and AD
    '''
    #Age total e MMSE total
    mmse_pred=df_total.MMSE_pred
    mmse_test=df_total.MMSE_test
    age_pred=df_total.Age_pred
    age_test=df_total.Age_test
    age_pred_ctrl=df_CTRL.Age_CTRL_pred
    age_pred_ad=df_CTRL.Age_AD_pred
    age_test_ctrl=df_CTRL.Age_CTRL_test
    age_test_ad=df_CTRL.Age_AD_test
    age_pred_ad_total=df_AD_total.Age_AD_total_pred
    age_test_ad_total=df_AD_total.Age_AD_total_test

    MAE_mmse=metrics.mean_absolute_error(mmse_test, mmse_pred)
    MAE_age=metrics.mean_absolute_error(age_test, age_pred)
    MAE_age_ctrl=metrics.mean_absolute_error(age_test_ctrl, age_pred_ctrl)
    MAE_age_ad=metrics.mean_absolute_error(age_test_ad, age_pred_ad)
    MAE_age_ad_total=metrics.mean_absolute_error(age_test_ad_total, age_pred_ad_total)
    RMSE_mmse=np.sqrt(metrics.mean_squared_error(mmse_test, mmse_pred))
    RMSE_age=np.sqrt(metrics.mean_squared_error(age_test, age_pred))
    RMSE_age_ctrl=np.sqrt(metrics.mean_squared_error(age_test_ctrl, age_pred_ctrl))
    RMSE_age_ad=np.sqrt(metrics.mean_squared_error(age_test_ad, age_pred_ad))
    RMSE_age_ad_total=np.sqrt(metrics.mean_squared_error(age_test_ad_total, age_pred_ad_total))

    print(f'MAE AGE: {MAE_age}; RMSE AGE: {RMSE_age}')
    print(f'MAE MMSE: {MAE_mmse}; RMSE MMSE: {RMSE_mmse}')
    print(f'MAE AGE CTRL: {MAE_age_ctrl}; RMSE AGE CTRL: {RMSE_age_ctrl}')
    print(f'MAE AGE AD: {MAE_age_ad}; RMSE AGE AD: {RMSE_age_ad}')
    print(f'MAE AGE AD TOTAL: {MAE_age_ad_total}; RMSE AGE AD TOTAL: {RMSE_age_ad_total}')

    # Box plot
    plt.figure()
    boxplot = df_diff_age.boxplot(column=['Diff_Age'], by='DX_GROUP', showfliers=False)
    boxplot.set_title('Box plot of age difference ')
    boxplot.get_figure().suptitle('');
    boxplot.set_ylabel('Diff_Age')
    boxplot.set_xticklabels(labels=boxplot.get_xticklabels());
    plt.show()

    # Box plot
    plt.figure()
    boxplot = df_diff_age_model.boxplot(column=['Diff_Age'], by='DX_GROUP', showfliers=False)
    boxplot.set_title('Box plot of age difference model')
    boxplot.get_figure().suptitle('');
    boxplot.set_ylabel('Diff_Age')
    boxplot.set_xticklabels(labels=boxplot.get_xticklabels());
    plt.show()

    #Permutation test age
    permutation_age(df_CTRL, Nperm=1000)
    #Permutation correcly and wrongly predicted
    permutation(df_AD, Nperm=1000, feature='Age_test')
    permutation(df_AD, Nperm=1000, feature='MMSE_test')
    permutation(df_ctrl, Nperm=1000, feature='Age_test')
