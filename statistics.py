""" Statistics analysis """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from input_dati import read_dataset, import_csv
SOGLIA = 0.387
#pylint: disable=invalid-name
#pylint: disable=line-too-long

def normalize(x):
    """
    Normalize the intensity of every pixel in the image
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
    Calculate Dice index for a single image
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
    dice_coef = intersection / (np.sum(pred) + np.sum(true))
    return dice_coef

def roc_curve(xtest, ytest, model):
    """
    Display ROC curve and calculate AUC
    :Parameters:
        xtest: 4D np.array
            array containg test images
        ytest: 2D np.array
            array containing test labels
    :Returns:
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

def plot_cv_roc(X, y, classifier, n_splits=5):
    """
    Implement k-fold-CV with k=n_splits, plot the ROC curves for each k fold
    and their average. It also display the corresponding AUC values and
    the standard deviation over the k folders.
    :Parameters:
        X : 4D np.array
            Array containing the images
        y : 1D np.array
            Array containing labels
        classifier : Keras.model
            The CNN model
        n_splits : int
            Number of folders. Default=5
    """

    model = classifier
    try:
        y = y.to_numpy()
        X = X.to_numpy()
    except AttributeError:
        pass

    cv = StratifiedKFold(n_splits)

    tprs = [] #True positive rate
    aucs = [] #Area under the ROC Curve
    trhs = [] #thresholds
    interp_fpr = np.linspace(0, 1, 100)
    plt.figure()
    i = 0
    for train, test in cv.split(X, y):

        y_score = model.predict(X[test])
        fpr, tpr, thresholds = metrics.roc_curve(y[test], y_score)
        #print(f"{fpr} - {tpr} - {thresholds}\n")
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        interp_threshold = np.interp(interp_fpr, fpr, thresholds)
        tprs.append(interp_tpr)
        trhs.append(interp_threshold)
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
    mean_trh = np.mean(trhs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(interp_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(interp_fpr, mean_tpr, color='b',
          label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
          lw=2, alpha=.8)
    plt.plot(interp_fpr, mean_trh, color='c',
          label='threshold',
          lw=2, alpha=.8)
    print('soglie')
    print(mean_trh)
    print('True positive')
    print(mean_tpr)

    j=0
    k=0
    while (k==0):
        if ((mean_tpr[j]> 0.76) and (mean_tpr[j] < 0.77)):
            indice=j
            k=k+1
        j=j+1

    plt.axvline(interp_fpr[indice], linestyle='--', color='green')
    print(f'La soglia è: {mean_trh[indice]}')
    print(f'FPR: {interp_fpr[indice]}')
    print(f'sensitività: {mean_tpr[indice]}')
    print(f'specificità: {1 - interp_fpr[indice]}')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                  label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC',fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()

def dataframe_test(xtest,ytest, agetest, mmsetest, age_max, mmse_max, model, agemodel, mmsemodel):
    '''
    Create the dataframes containig labels, age, MMSE and a Confront_predizione
    for test's images. Confronto_predizione is an array that is 1 if the prediction
    of the model is correct and 0 if it is wrong.
    :Parameters:
        xtest : 4D array
            Array containg the test's images
        ytest : 1D array
            Array containg the labels of test's images
        agetest : Tupla
            Tupla containing the age relative to test's images
        mmsetest : Tupla
            Tupla containg the MMSE relative to test's images
        age_max : Double
            maximum age relative to test's images
        mmse_max : Double
            maximum mmse relative to test's images
        model : Keras model
            base model containing CNN layers
        agemodel : Keras model
            model used to predict age
        mmsemodel : Keras model
            model used to predict mmse
    :Returns:
        dataFrame : pandas dataframe
            Dataframe contining the feautures of all test's images
        dataFrame_AD : pandas dataframe
            Dataframe contaning the features of test images belongig to AD category
        dataFrame_CTRL :
            Dataframe contaning the feature of test images belongig to CTRL category
    '''

    X_test = tf.expand_dims(xtest, axis=-1)
    y_pred_test = model.predict(X_test)
    age_pred = agemodel.predict(X_test)
    mmse_pred = mmsemodel.predict(X_test)

    y_pred_test = np.squeeze(y_pred_test)
    age_pred = np.squeeze(age_pred)
    mmse_pred = np.squeeze(mmse_pred)

    mmsetest = np.array(mmsetest)
    age_pred= age_pred*age_max
    mmse_pred=mmse_pred*mmse_max

    print(f' y test : {np.shape(ytest)}')
    print(f' y pred  : {np.shape(y_pred_test)}')
    print(f'Age pred :{np.shape(age_pred)}')
    print(f'mmse pred: {np.shape(mmse_pred)}')

    y_conf = np.empty(shape=len(ytest), dtype=bool)

    for i in range(0,len(y_pred_test)):
        y_conf[i] = (ytest[i] == (y_pred_test[i]>SOGLIA))

    y_Conf = np.empty(shape=len(ytest), dtype=int)
    for i in range(0, len(y_conf)):
        if y_conf[i] == True:
            y_Conf[i]=1
        if y_conf[i] == False:
            y_Conf[i]=0

    d = {'labels_test': ytest, 'Confronto_predizione': y_Conf, 'Age_test' : agetest, 'Age_pred' : age_pred, 'MMSE_test' : mmsetest, 'MMSE_pred' : mmse_pred }

    dataFrame = pd.DataFrame(data=d)
    dataFrame_AD = dataFrame[dataFrame.labels_test == 1]
    dataFrame_CTRL = dataFrame[dataFrame.labels_test == 0]

    return dataFrame, dataFrame_AD, dataFrame_CTRL

def correlation(df, dfAD, dfCTRL):
    '''
    Compute correlations between features in dataframes containing the features
    of test images and their predicted category.
    :Parameters:
        df : pandas dataframe
            Dataframe containing the features of all the test images
        dfAD : pandas dataframe
            Dataframe containing the features of the test images belongig to AD category
        dfCTRL: pandas dataframe
            Dataframe containing the features of the test images belongig to CTRL category
    :Returns:
        None
    '''
    #correlazione con tutti i casi di test, quindi AD e controllo
    print(df.drop('labels_test', axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe total')
    sns_heatmap=sns.heatmap(df.drop('labels_test', axis=1).corr())
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #correlazione con AD di test
    print(dfAD.drop('labels_test', axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe AD')
    sns_heatmap=sns.heatmap(dfAD.drop('labels_test', axis=1).corr())
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    #correlazione con CTRL di test
    print(dfCTRL.drop('labels_test', axis=1).corr())
    #heatmap
    plt.figure()
    plt.title('Heatmap correlations dataframe CTRL')
    sns_heatmap=sns.heatmap(dfCTRL.drop('labels_test', axis=1).corr())
    sns_heatmap.set_xticklabels(labels=sns_heatmap.get_xticklabels(), rotation=20)
    plt.show()

    return None

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

def scatter_plot(data_frame):
    """
    Display scatter plots of age and MMSE
    :Parameters:
        data_frame : Pandas DataFrame
            DataFrame containing test images features
    :Returns:
        None
    """
    color = data_frame.labels_test.apply(lambda x:'blue' if x == 0 else 'red')

    #in blu quelli controllo e in rosso quelli AD
    ax = data_frame.plot(x='Age_test', y='Age_pred', kind='scatter', color=color)
    ax.grid()

    # build the legend
    red_patch = mpatches.Patch(color='red', label='AD')
    blue_patch = mpatches.Patch(color='blue', label='CTRL')
    patches = [red_patch, blue_patch]
    legend = ax.legend(handles=patches,loc='upper left')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title('Scatter plot age')
    plt.show()

    ax = data_frame.plot(x='MMSE_test', y='MMSE_pred', kind='scatter', color=color, label=True)
    ax.grid()

    # build the legend
    red_patch = mpatches.Patch(color='red', label='AD')
    blue_patch = mpatches.Patch(color='blue', label='CTRL')
    patches = [red_patch, blue_patch]
    legend = ax.legend(handles=patches,loc='upper left')

    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.title('Scatter plot mmse')
    plt.show()

if __name__=='__main__':

    dataset_path_AD_ROI = "AD_CTRL/AD_ROI_TH"
    dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROI_TH"
    dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

    # Import csv data
    df, head, dict_age, dict_mmse = import_csv(dataset_path_metadata)
    features = ['DXGROUP', 'ID', 'AGE', 'MMSE']
    print(df[features])

    # import images, labels, file names, age and mmse
    X, Y, fnames_AD, fnames_CTRL, file_id, age, mmse = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI,dict_age, dict_mmse , str_1='1', str_2='.')

    age = np.array(age)
    mmse = np.array(mmse)
    AGE_MAX=np.max(age)
    MMSE_MAX=np.max(mmse)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=11)
    _, age_test, _, mmse_test = train_test_split(age, mmse, test_size=0.15, random_state=11)
    print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
    print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')

    model = tf.keras.models.load_model("Modelli/3d_CNN_15_50_100_Hipp.h5")
    model.summary()

    age_model = tf.keras.models.load_model("Modelli/3d_age_regression_15_50_100_MAE_MSE_64_128_128_64_tun.h5")
    age_model.summary()

    mmse_model = tf.keras.models.load_model("Modelli/3d_mmse_regression_15_50_100_MAE_MSE_64_128_128_64_tun.h5")
    mmse_model.summary()

    auc = roc_curve(X_test, Y_test, model)

    plot_cv_roc(X,Y, model, 5, scaler=None)

    #Calcolo indice di Dice

    xtrain = np.expand_dims(X_train, axis=-1)
    ytrain = np.expand_dims(Y_train, axis=-1)
    #print(Y_train[idx].shape, ytrain.shape)

    ypred = model.predict(xtrain).squeeze()>SOGLIA
    ytrue = Y_train.squeeze()

    dice_value = dice(ypred, ytrue)
    print(f'Indice di DICE:{dice_value}')

    #Accuracy
    accuracy = metrics.accuracy_score(ytrue, ypred)
    print(f'Accuracy:{accuracy}')

    df, df_AD, df_CTRL=dataframe_test(X_test, Y_test, age_test, mmse_test, AGE_MAX, MMSE_MAX, model, age_model, mmse_model)
    print(df.head())
    correlation(df, df_AD, df_CTRL)
    permutation(df, Nperm=1000, feature='Age_test')
    permutation(df, Nperm=1000, feature='MMSE_test')
    scatter_plot(df)

    '''
    Compute MAE e RMSE for MMSE and AGE
    '''
    mmse_pred=df.MMSE_pred
    age_pred=df.Age_pred

    MAE_mmse=metrics.mean_absolute_error(mmse_test, mmse_pred)
    MAE_age=metrics.mean_absolute_error(age_test, age_pred)

    RMSE_mmse=np.sqrt(metrics.mean_squared_error(mmse_test, mmse_pred))
    RMSE_age=np.sqrt(metrics.mean_squared_error(age_test, age_pred))

    print(f'MAE AGE: {MAE_age}; RMSE AGE: {RMSE_age}')
    print(f'MAE MMSE: {MAE_mmse}; RMSE MMSE: {RMSE_mmse}')
