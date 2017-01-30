import pandas as pd
import data_creation as dc
import copy


hdf_file = '../data/subject_data.h5'


def df_concat(sensor_num):
    df_list = []
    for i in range(1, 9):
        print('Subject %d' % i)
        data = dc.SubjectData(hdf_file, i, sensor_num)
        data.artefact_row_extraction()
        data.group_labels()
        data.label_raw_data()
        df_list.append(data.raw_df)

    all_df = pd.concat(df_list)
    print('\nWriting data to file\n')
    all_df.to_csv('../data/raw_sensor_%s.csv' % sensor_num,
                  index=False)
    print('\nraw_sensor_%s.csv written to file.\n' % sensor_num)
    return None

if __name__ == '__main__':
    df_concat(7)
    df_concat(13)
