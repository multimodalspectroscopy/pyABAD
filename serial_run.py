import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyabad.machine_learning as ML

def parallel_split(df, n_workers):
    x = [df.shape[0]//n_workers*k for k in range(n_workers)]
    x.append(df.shape[0])
    return [df[x[k]:x[k+1]] for k in range(n_workers)]


if __name__ == '__main__':
    from datetime import datetime
    df_7 = pd.read_csv('data/raw_sensor_7.csv')
    #df_13 = pd.read_csv('../data/raw_sensor_13.csv')
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    target_names = ['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown', 'Ambient Light', 'Torch Light']
    features_7 = ML.feature_creation(df_7)
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    features_7.to_csv('../data/df_7/engineeredfeatures%s.csv' %
                      (date), index=False)
    #test_train_data = ML.test_train_split(features_7)
    #fpr, tpr, auroc, n_classes = ML.classification(test_train_data[0])
    #ML.ROC_plot(fpr, tpr, auroc, n_classes, date, target_names)
    """
    features_13 = feature_creation(df_13)
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    features_13.to_csv('../data/df_13/engineeredfeatures%s.csv' %
                       (date), index=False)
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
    """