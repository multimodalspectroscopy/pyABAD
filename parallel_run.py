import pandas as pd
import numpy as np
import pyabad.machine_learning as ML
from multiprocessing import Pool, TimeoutError
from datetime import datetime
import os
DATAPATH = os.path.join('.','data')

def parallel_split(df, n_workers):
    x = [df.shape[0]//n_workers*k for k in range(n_workers)]
    x.append(df.shape[0])
    return [df[x[k]:x[k+1]] for k in range(n_workers)]


def output_join(list_of_dataframes):
    return pd.concat(list_of_dataframes).sort_index()


def parallel_batch(df, n_workers=4):
    n_workers = n_workers

    pooledData = parallel_split(df, n_workers)
    results = []
    with Pool(processes=n_workers) as pool:
        results = pool.map(ML.feature_creation, pooledData)
        #results.append(i)

    pool.join()
    features = output_join(results)
    return features

def configure_pipeline(sensor_num, n_workers=4, light_split=False, motion_split=False):
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    config = {"n_workers": n_workers,
              "light_split": light_split,
              "motion_split": motion_split,
              "date":date,
              "sensor_num":sensor_num,
              'target_name':['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown', 'Ambient Light', 'Torch Light']}
    return config

def pipeline(df, config):
    features = parallel_batch(df, config['n_workers'])
    features.to_csv(os.path.join(DATAPATH,
                                 'df_%s' % config['sensor_num'],
                                 'parallel_features_%s.csv' % config['sensor_num']))
    split_data = ML.test_train_split(features)
    fpr, tpr, auroc, n_classes, clf = ML.classification(split_data[0])
    ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config['target_names'], config['sensor_num'], 'Training')
    fpr, tpr, auroc, n_classes = ML.final_test(split_data[1],clf)
    ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config['target_names'], config['sensor_num'], 'Test')


if __name__=='__main__':
    df_7 = pd.read_csv('data/raw_sensor_7.csv')
    features_7 = parallel_batch(df_7, 7)
    features_7.to_csv('~/Desktop/parallel_features_7.csv')

    #df_13 = pd.read_csv('data/raw_sensor_13.csv')
    #features_13 = parallel_batch(df_13, 7)
    #features_13.to_csv('~/Desktop/parallel_features_13.csv')


