import pandas as pd
import data_creation as dc
import copy


HDF_FILE = '../../data/subject_data.h5'


def df_concat(sensor_num, subject_number):
    """
    Crate a pandas dataframe for each subject and then concatenate to generate
    appropriate data structures for use in machine learning.
    Inputs:
    :param sensor_num: The sensor number - either the mac subject number or a
    list of numbers.
    :type sensor_num: Union(int, float, list)

    """
    df_list = []

    if type(subject_number) == int:
        subject_list = list(range(1, subject_number))
    elif type(subject_number) == float:
        subject_list = list(range(1, int(round(subject_number))))
    elif type(subject_number) == list:
        subject_list = subject_number
    else:
        print('Not a valid sensor number specification. See help(df_concat)')

    for i in subject_list:
        print('Subject %d' % i)
        data = dc.SubjectData(HDF_FILE, i, sensor_num)
        data.artefact_row_extraction()
        data.group_labels()
        data.label_raw_data()
        df_list.append(data.raw_df)

    all_df = pd.concat(df_list)
    print('\nWriting data to file\n')
    all_df.to_csv('../../data/raw_sensor_%s.csv' % sensor_num,
                  index=False)
    print('\nraw_sensor_%s.csv written to file.\n' % sensor_num)
    return None


if __name__ == '__main__':
    df_concat(7)
    df_concat(13)
