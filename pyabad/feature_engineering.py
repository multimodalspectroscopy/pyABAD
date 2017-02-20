import pandas as pd

# Local module import
from features import SampEn, AUC, autocorr, average_PSD

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
    progress = percentage_coroutine(len(y))
    next(progress)
    features = pd.DataFrame()
    features['Artefact'] = y.values
    features['Subject'] = groups.values
    print('Calculating AUC')
    features['AUC'] = AUC(X, progress)
    progress = percentage_coroutine(len(y))
    next(progress)
    print('Calculating PSD')
    features['PSD'] = X.apply(trace_progress(
        average_PSD, progress=progress), raw=True, axis=1)
    progress = percentage_coroutine(len(y))
    next(progress)
    print('Calculating AutoCorr')
    features['AutoCorr'] = X.apply(trace_progress(
        lambda x: x.autocorr(lag=1), progress=progress), axis=1)
    progress = percentage_coroutine(len(y))
    next(progress)
    print('Calculating SampEn')
    features['SampEn'] = X.apply(trace_progress(
        SampEn, progress=progress), raw=True, axis=1)

    return features