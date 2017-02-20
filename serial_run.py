import pandas as pd
import pyabad.machine_learning as ML
from datetime import datetime
import os
DATAPATH = os.path.join('.','data')


def artefact_name(x):
    return {"light":"Light","motion":"Motion",None:"All"}.get(x, 'INVALID')


def configure_pipeline(sensor_num, artefact_type=None ):
    dt = datetime.now()
    date = "".join(filter(lambda char: char.isdigit(), str(dt)))[:14]
    config = {"artefact_type": artefact_type,
              "date":date,
              "sensor_num":sensor_num,
              'target_name':['Control', 'Horizontal', 'Vertical', 'Pressure',
                    'Frown', 'Ambient Light', 'Torch Light']}
    return config

def pipeline(df, config):
    features = ML.feature_creation(ML.motion_light_split(df, config['artefact_type']))
    artefact = artefact_name(config['artefact_type'])
    features.to_csv(os.path.join(DATAPATH,
                                 'df_%s' % config['sensor_num'],
                                 'serial_features_%s_%s.csv' % (config['sensor_num'], artefact)))
    split_data = ML.test_train_split(features)
    fpr, tpr, auroc, n_classes, clf = ML.classification(split_data[0])
    ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config['target_names'], config['sensor_num'], 'Training')
    fpr, tpr, auroc, n_classes = ML.final_test(split_data[1],clf)
    ML.ROC_plot(fpr, tpr, auroc, n_classes, config['date'], config['target_names'], config['sensor_num'], 'Test')

    return True


if __name__=='__main__':
    df_7 = pd.read_csv('data/raw_sensor_7.csv')
    df_13 = pd.read_csv('data/raw_sensor_13.csv')

    config = configure_pipeline(7)
    pipeline(df_7, config)

    config = configure_pipeline(7, artefact_type='light')
    pipeline(df_7, config)

    config = configure_pipeline(7, artefact_type='motion')
    pipeline(df_7, config)