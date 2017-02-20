import os
import argparse
from pyabad.data_creation import hdf_creation as hdf
from pyabad import utility


def file_walk(hdf_fname, rootDir, create_file=True):
    """
    Walk through a directory of files to create the necessary
    HDF5 file.
    Inputs:
    :param hdf_fname: The expected HDF5 file name and location
    :type hdf_fname: str
    :param rootDir: The directory containing the CSV files.
    : type rootDir: str
    :param create_file: Argument as to whether the HDF5 file needs creating.
    :type create_file: bool
    """
    print('Initiating File Walk')
    for dirName, subdirList, fileList in os.walk(rootDir):
        if 'SUBJECT' in dirName:
            print('Found directory: %s' % dirName)
            for fname in fileList:
                print('\t Processing %s' % fname)
                if 'time' in fname:
                    time_file = os.path.join(dirName, fname)
                elif ('conc_7' in fname):
                    conc_7 = os.path.join(dirName, fname)
                elif ('conc_13' in fname):
                    conc_13 = os.path.join(dirName, fname)
                elif ('raw_7' in fname):
                    raw_7 = os.path.join(dirName, fname)
                elif ('raw_13' in fname):
                    raw_13 = os.path.join(dirName, fname)
                else:
                    pass

            HDF = hdf.HDF5_Subject(hdf_fname, int(dirName[-3:]),
                                   raw_7, conc_7, time_file, 7)
            if create_file:
                try:
                    open(hdf_fname, 'x')
                    HDF.file_creation()
                    print("\n\t File created. \n")
                    create_file = False
                except:
                    print("\n File already exists.")

            HDF.print_file()
            HDF.append_time_data()
            HDF.append_raw_data()
            HDF.append_conc_data()

            del HDF

            HDF = hdf.HDF5_Subject(hdf_fname, int(dirName[-3:]),
                                   raw_13, conc_13,
                                   time_file, 13)
            if create_file:
                try:
                    open(hdf_fname, 'x')
                    HDF.file_creation()
                    print("\n\t File created. \n")
                    create_file = False
                except FileExistsError:
                    print("\n File already exists.")

            HDF.print_file()
            HDF.append_time_data()
            HDF.append_raw_data()
            HDF.append_conc_data()
        else:
            print('No subject data')
            pass


def main():
    parser = argparse.ArgumentParser(description='Walk files to create HDF5')
    parser.add_argument(
        'hdf_fname', type=str, help="Path to HDF file", metavar='HDF_FILE')
    parser.add_argument(
        'rootDir', type=str, help="Root directory to walk", metavar='ROOT_DIR')
    args = parser.parse_args()
    print('\n%s\n%s' % (args.hdf_fname, args.rootDir))
    file_walk(args.hdf_fname, args.rootDir)


if __name__ == '__main__':
    main()
