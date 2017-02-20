import pandas as pd
from sklearn import preprocessing
import numpy as np
# Local module import
from .features import SampEn, AUC, autocorr, average_PSD

# Function to trace feature progress.
def percentage_coroutine(to_process, print_on_percent=0.05):
    print("Starting progress percentage monitor \n")

    processed = 0
    count = 0
    print_count = to_process * print_on_percent
    while True:
        yield
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed) / float(to_process))

            print("{:.0%} finished".format(pct))


def trace_progress(func, progress=None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf

def feature_engineering(X, y, groups):
    co2 = percentage_coroutine(len(y))
    next(co2)
    features = pd.DataFrame()
    features['Artefact'] = y.values
    features['Subject'] = groups.values
    print('Calculating AUC')
    auc=X.apply(trace_progress(
        np.trapz, progress=co2), raw=True, axis=1)
    #scaler = preprocessing.MinMaxScaler()
    #print('Normalising AUC')
    #scaled_auc = scaler.fit_transform(auc.reshape(-1, 1))
    features['AUC'] = auc

    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating PSD')
    features['PSD'] = X.apply(trace_progress(
        average_PSD, progress=co2), raw=True, axis=1)

    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating AutoCorr')
    features['AutoCorr'] = X.apply(trace_progress(
        lambda x: x.autocorr(lag=1), progress=co2), axis=1)

    co2 = percentage_coroutine(len(y))
    next(co2)
    print('Calculating SampEn')
    features['SampEn'] = X.apply(trace_progress(
        SampEn, progress=co2), raw=True, axis=1)

    features.index=X.index
    return features


