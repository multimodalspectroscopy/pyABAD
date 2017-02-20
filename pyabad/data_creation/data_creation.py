import numpy as np
import csv
import tables as tb
import itertools
import warnings
import argparse
import pandas as pd
import os

class SubjectData:
    """A class to collate, clean and label artefact data.
    It takes in the 3 CSV files - raw, conc and time and outputs
    Panda dataframes that are labelled. CSV files are stored inside
    a HDF5 file.
    """

    def __init__(self, hdf_file, SUBJECT_NUMBER, sensor_number):
        self.hdf = hdf_file
        self.num = SUBJECT_NUMBER
        self.sensor = sensor_number
        self.rowNums = []
        self.labels = []
        self.raw_df = None
        self.conc_df = None

    def artefact_row_extraction(self):
        with tb.open_file(self.hdf, mode='r') as h5file:
            table = h5file.get_node('/SUBJECT_%03d/TIME_DATA' % (self.num))
            self.rowNums = [idx for idx, x in enumerate(table.iterrows())
                            if x['artefact'] != b'']

            if len(self.rowNums) != 25:
                warnings.warn('Incorrect number of artefact markers \\'
                              'detected. \n\t %d found when there should \\'
                              'be 25' % (len(self.rowNums)))

    def group_labels(self):

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        group_lengths = [y - x for x, y in pairwise(self.rowNums)]
        # TODO : make the below much neater
        group_order = [item for sublist in
                       [x for n in range(1, 7)
                        for x in zip(itertools.repeat(0, 2),
                                     itertools.repeat(n, 2))]
                       for item in sublist]
        self.labels = [ii for grp, n in zip(
            group_order, group_lengths) for ii in itertools.repeat(grp, n)]

    def label_raw_data(self):
        wavelengths = []
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'wavelengths.txt'), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                wavelengths.append(row[0])
        with tb.open_file(self.hdf, mode='r') as h5file:
            raw_array = h5file.get_node(
                '/SUBJECT_%03d/SENSORS/SENSOR_%d/raw_data_%d'
                % (self.num, self.sensor, self.sensor)).read()
        self.raw_df = pd.DataFrame(
            data=raw_array[self.rowNums[0]:self.rowNums[-1]],
            columns=wavelengths, copy=True)
        self.raw_df['Artefact'] = self.labels
        self.raw_df['Subject'] = self.num

    def label_conc_data(self):
        with tb.open_file(self.hdf, mode='r') as h5file:
            conc_array = h5file.get_node(
                '/SUBJECT_%03d/SENSORS/SENSOR_%d/conc_data_%d'
                % (self.num, self.sensor, self.sensor)).read()
        self.conc_df = pd.DataFrame(
            data=conc_array[self.rowNums[0]:self.rowNums[-1]],
            columns=['HbO2', 'HHb', 'oxCCO', 'HbT'],
            copy=True)
        self.conc_df['Artefact'] = self.labels
        self.conc_df['Subject'] = self.num


def main():
    parser = argparse.ArgumentParser(description='Receive raw, conc and time\\'
                                     ' series files for creating a subject \\'
                                     'artefact data file')
    parser.add_argument('hdf_file',
                        type=str,
                        help="Path to HDF file",
                        metavar='HDF_FILE')
    parser.add_argument('subject_number',
                        type=int,
                        help="Subject number e.g. 1,2,...",
                        metavar='SUBJ_NUM')
    parser.add_argument('sensor_number',
                        type=int,
                        help="Sensor number - 7 or 13",
                        metavar='SENS_NUM',
                        choices=[7, 13])

    args = parser.parse_args()

    data = SubjectData(args.hdf_file, args.subject_number, args.sensor_number)

    data.artefact_row_extraction()
    data.group_labels()
    data.label_raw_data()

    return(data)


if __name__ == '__main__':
    data = main()
