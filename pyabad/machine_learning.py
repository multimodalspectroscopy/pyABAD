import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import cycle
import math
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import svm

import scipy.fftpack as sp_fft
import scipy

from sampen import sampen2

# Function to trace feature progress.
def percentage_coroutine(to_process, print_on_percent = 0.05):
    print("Starting progress percentage monitor \n")

    processed = 0
    count = 0
    print_count = to_process*print_on_percent
    while True:
        yield
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed)/float(to_process))

            print("{:.0%} finished".format(pct))

def trace_progress(func, progress = None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf

def ApEn(U, m, r):
    """
    Function for approximate entropy as per Wikipedia
    :param U: Time series of data
    :param m: Length of compared run of data (typically 2)
    :param r: Filtering level
    :return: Approximate entropy value for the time series
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def SampEn(X, verbose=False):
    """
    Implementation of Sample Entropy using sampen package
    :param X: Numpy array of spectra
    :return: Sample Entropy float
    """
    sampen = sampen2(X, r=10)
    if verbose:
        print(sampen)
    return sampen[2][1]



def AUC(X, coroutine):
    """
    Function to return the area under a spectra
    :param X: Pandas dataframe of raw sensor data.
    :return: Pandas series of AUC for each spectra.
    """
    auc = X.apply(trace_progress(np.trapz, progress=coroutine),raw=True,axis=1)
    scaler = preprocessing.MinMaxScaler()
    print('Normalising AUC')
    return scaler.fit_transform(auc.reshape(-1, 1))


def average_PSD(X_origin, plot=False):
    """
    Power spectral density of the bottom 1% of frequencies
    as a fraction of the whole.
    :param X_origin: Pandas dataframe of spectra data
    :param plot: Boolean stating whether FFT is to be plotted or not
    :return: x,y of the fourier if plotting, PSD if not
    """
    X = copy.deepcopy(X_origin)
    N = X.shape[0]
    yf = sp_fft.rfft(X)
    xf = sp_fft.rfftfreq(N)
    psd = np.abs(yf)**2
    PSD = metrics.auc(xf[:N//100], psd[:N//100])/metrics.auc(xf, psd)
    if plot:
        return xf, yf
    else:
        return PSD

def autocorr(X):
    """
    Function to find autocorrelation values of spectra data.
    :param X: Dataframe of raw spectra data
    :return: Time series of autocorrelation data
    """
    return X.apply(lambda x: x.autocorr(lag=1), axis=1)


def feature_engineering(X,y,groups):
    co2 = percentage_coroutine(len(y))
    next(co2)
    features = pd.DataFrame()
    features['Artefact'] = y.values
    features['Subject'] = groups.values
    print('Calculating AUC')
    features['AUC']= AUC(X, co2)
    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating PSD')
    features['PSD'] = X.apply(trace_progress(average_PSD, progress=co2), raw=True, axis=1)
    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating AutoCorr')
    features['AutoCorr'] = X.apply(trace_progress(lambda x: x.autocorr(lag=1), progress=co2), axis=1)
    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating SampEn')
    features['SampEn'] = X.apply(trace_progress(SampEn, progress=co2),raw=True,axis=1)

    return features

def data_separation(df):
    """
    Funciton to split a dataframe up into X, y and groups
    :param df: Dataframe to split. Must have Artefacts and Subject columns
    :return: (X, y, groups)
    """
    X = df.drop(['Artefact', 'Subject'], axis=1)
    y = df['Artefact']
    groups = df['Subject']
    return(X, y, groups)

def feature_creation(df):
    X, y, groups = data_separation(df)

    features = feature_engineering(X, y, groups)

    return features


def test_train_split(df):
    """
    Function to split a dataframe of engineered features by subject and output required Dataframes.
    :param df: Input dataframe from csv file.
    :return: list of dicts of form [[train_data],[test_data]], with data being X,  y, groups.
    """
    rand_state = np.random.randint(100)
    X, y, groups = data_separation(df)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=10, random_state=rand_state).split(X, y, groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_true = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    return [{'X':X_train, 'y':y_train, 'groups': groups_train},{'X': X_test, 'y':y_true, 'groups':groups_test}]


def motion_light_split(features, *, light=False, motion=False):
    if light:
        new_features = features.drop(features[features['Artefact'].isin({1, 2, 3, 4})].index, axis=0)
    elif motion:
        new_features = features.drop(features[features['Artefact'].isin({5, 6})].index, axis=0)
    else:
        print("No split required. \n")
        return None

    return new_features


def roc_area(clf, X_test, y_test, n_classes):
    """
    Method of averaging the AUROC scores for each class
    :param clf: fitted classifier in question
    :param X_test: test input data
    :return: ROCAUC
    """
    y_score = clf.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return fpr,tpr,roc_auc

def classification(train):
    """
    Function to classify the training data output from train_test_split
    :param test_train_data: Output from test_train_split function
    :return: False positive rate, true positive rate and AUROC
    """


    rand_state = np.random.randint(100)

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=10, random_state=rand_state).split(train['X'],
                                                                                             train['y'],
                                                                                             train['groups']))

    # Binarize the training data
    y = preprocessing.label_binarize(train['y'], classes=list(set(train['y'])))
    n_classes = y.shape[1]

    classifier = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=rand_state))

    classifier.fit(train['X'].iloc[train_idx], y[train_idx])

    fpr, tpr, auroc = roc_area(classifier, train['X'].iloc[test_idx], y[test_idx], n_classes)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auroc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, auroc, n_classes

def ROC_plot(fpr,tpr,auroc, n_classes, datetime, target_names):
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(auroc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(auroc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(sns.color_palette("husl", n_classes))


    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(target_names[i], auroc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of AUROC scores for each class of Artefact')
    plt.legend(loc="lower right")
    plt.savefig("../figures/ROC_Curve_%s"%datetime)
    #plt.show()



if __name__=='__main__':
    from datetime import datetime
    df_7 = pd.read_csv('../data/raw_sensor_7.csv')
    df_13 = pd.read_csv('../data/raw_sensor_13.csv')
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    target_names = ['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown','Ambient Light', 'Torch Light']
    features_7 = feature_creation(df_7)
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    features_7.to_csv('../data/df_7/engineeredfeatures%s.csv'%(date), index=False)
    test_train_data = test_train_split(features_7)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    ROC_plot(fpr,tpr,auroc,n_classes,date, target_names)

    features_13 = feature_creation(df_13)
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    features_13.to_csv('../data/df_13/engineeredfeatures%s.csv' % (date), index=False)
    test_train_data = test_train_split(features_13)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    ROC_plot(fpr, tpr, auroc, n_classes, date, target_names)


    motion_7 = feature_creation(motion_light_split(df_7, motion=True))
    light_7 = feature_creation(motion_light_split(df_7, light=True))
    test_train_data = test_train_split(motion_7)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    motion_names = ['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown']
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    ROC_plot(fpr, tpr, auroc, n_classes, date, motion_names)

    test_train_data = test_train_split(light_7)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    light_names = ['Control', 'Ambient Light', 'Torch Light']
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    ROC_plot(fpr, tpr, auroc, n_classes, date, light_names)

    motion_13 = feature_creation(motion_light_split(df_13, motion=True))
    light_13 = feature_creation(motion_light_split(df_13, light=True))

    test_train_data = test_train_split(motion_13)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    motion_names = ['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown']
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    ROC_plot(fpr, tpr, auroc, n_classes, date, motion_names)

    test_train_data = test_train_split(light_13)
    fpr, tpr, auroc, n_classes = classification(test_train_data[0])
    light_names = ['Control', 'Ambient Light', 'Torch Light']
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    ROC_plot(fpr, tpr, auroc, n_classes, date, light_names)
