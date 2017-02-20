import scipy.fftpack as sp_fft
import numpy as np
import copy
from sampen import sampen2
from sklearn import metrics


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
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) /
             (N - m + 1.0) for x_i in x]
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

def trace_progress(func, progress=None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

def AUC(X, coroutine):
    """
    Function to return the area under a spectra
    :param X: Pandas dataframe of raw sensor data.
    :return: Pandas series of AUC for each spectra.
    """
    auc = X.apply(trace_progress(
        np.trapz, progress=coroutine), raw=True, axis=1)
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
    PSD = metrics.auc(xf[:N // 100], psd[:N // 100]) / metrics.auc(xf, psd)
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