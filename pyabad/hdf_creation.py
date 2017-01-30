import tables as tb
import numpy as np
import csv
import argparse


class HDF5_Subject:
    """A class to convert collected artefact data into a single HDF5 file
    using PyTables. Imported files are expected to be of a CSV type.
    """

    def __init__(self, fname, subject_number, raw_file, conc_file, ts_file,
                 sensor_number):
        self.fname = fname
        self.num = subject_number
        self.raw = raw_file
        self.conc = conc_file
        self.ts = ts_file
        self.sensor = sensor_number

    def file_creation(self):
        with tb.open_file(self.fname, mode="w",
                          title="Set of subjects' data") as h5file:
            h5file.close()

    def append_subject(self):
        class TimeData(tb.IsDescription):
            timestamp = tb.StringCol(9)
            duration = tb.Float32Col()
            artefact = tb.StringCol(8)
        with tb.open_file(self.fname, mode="a") as h5file:
            subject = h5file.create_group("/",
                                          'SUBJECT_%03d' % (self.num),
                                          'Subject %d' % (self.num))

            h5file.create_table(subject, 'TIME_DATA',
                                TimeData, "Time Series Data")

            sensors = h5file.create_group(subject,
                                          'SENSORS',
                                          'Data from sensors 7 and 13')

            h5file.create_group(sensors, 'SENSOR_7', 'Sensor 7 data')

            h5file.create_group(sensors, 'SENSOR_13', 'Sensor 13 data')

    def print_file(self):
        with tb.open_file(self.fname, mode="r") as h5file:
            print("\nSUBJECT DATA HDF5\nsubject_data.h5 currently has form:\n")
            print(h5file)

    def append_time_data(self):
        path = '/SUBJECT_%03d' % (self.num)
        with tb.open_file(self.fname, mode="r+") as h5file:
            if not h5file.__contains__(path):
                print("\nCreating subject data %s\n" % (path[1:]))
                self.append_subject()
            table = h5file.get_node(path).TIME_DATA
            print('Table has %d rows.' % table.nrows)
            if table.nrows == 0:
                print("\nAppending Time Series data\n")
                entry = table.row
                fields = ['Timestamp', 'Artefact', 'Elapsed_Time']
                with open(self.ts, 'r') as csvfile:
                    reader = csv.DictReader(
                        csvfile, fieldnames=fields, delimiter=',')
                    for row in reader:
                        entry['timestamp'] = row['Timestamp']
                        entry['duration'] = row['Elapsed_Time']
                        entry['artefact'] = row['Artefact']
                        entry.append()
                table.flush()
                print("\nTime series appended.\n\t nrows = %d" % (table.nrows))
            else:
                print("\nTime Series Data already exists.\n")

    def append_raw_data(self):
        path = '/SUBJECT_%03d' % (self.num)
        with tb.open_file(self.fname, mode="r+") as h5file:
            if not h5file.__contains__(path):
                print("\nCreating subject data %s\n" % (path[1:]))
                self.append_subject()
            group = h5file.get_node(path + "/SENSORS/SENSOR_%d" % self.sensor)
            try:
                print("\nAttempting to store raw data\n")
                # Check for node to throw exception before loading array
                h5file.get_node(path + "/SENSORS/SENSOR_%d/raw_data_%d" %
                                (self.sensor, self.sensor))
                print("\nRaw data array already exists.\n")

            except:
                raw_array = np.genfromtxt(self.raw, delimiter=',')
                h5file.create_array(group, 'raw_data_%d' % (self.sensor),
                                    raw_array,
                                    "Raw data for sensor %d" % self.sensor)
                print("\nRaw data stored.\n")

    def append_conc_data(self):
        path = '/SUBJECT_%03d' % (self.num)
        with tb.open_file(self.fname, mode="r+") as h5file:
            if not h5file.__contains__(path):
                print("\nCreating subject data %s\n" % (path[1:]))
                self.append_subject()
            group = h5file.get_node(path + "/SENSORS/SENSOR_%d" % self.sensor)
            try:
                print("\nAttempting to store concentration data\n")
                # Check for node to throw exception before loading array
                h5file.get_node(path + "/SENSORS/SENSOR_%d/conc_data_%d" %
                                (self.sensor, self.sensor))
                print("\nConcentration data array already exists.\n")
            except:
                raw_array = np.genfromtxt(self.conc, delimiter=',')
                h5file.create_array(group, 'conc_data_%d' % (self.sensor),
                                    raw_array,
                                    "Concentration data for sensor %d"
                                    % self.sensor)
                print("\nConcentration data stored.\n")


def main():
    parser = argparse.ArgumentParser(description='Create HDF5 file')
    parser.add_argument(
        'fname', type=str, help="Path to HDF file", metavar='HDF_FILE')
    parser.add_argument(
        'raw', type=str, help="Path to raw data file", metavar='RAW_DATA')
    parser.add_argument(
        'conc', type=str, help="Path to conc data file", metavar='CONC_DATA')
    parser.add_argument(
        'time', type=str, help="Path to time series file", metavar='TS_DATA')
    parser.add_argument('number', help='Subject number e.g. 1,2,...',
                        action='store', type=int, metavar='SUBJECT_NUMBER')
    parser.add_argument('sensor_number', help='Sensor Number - 7 or 13',
                        action='store', type=int, metavar='SENSOR_NUMBER',
                        choices=[7, 13])
    parser.add_argument('-C', '--create', action='store_true',
                        help='Flag whether to create HDF5 file.')
    args = parser.parse_args()
    HDF5 = HDF5_Subject(args.fname, args.number, args.raw, args.conc,
                        args.time, args.sensor_number)
    if args.create:
        try:
            open(args.fname, 'x')
            HDF5.file_creation()
            print("\n\t File created. \n")
        except:
            print("\n File already exists.")
    HDF5.print_file()
    HDF5.append_time_data()
    HDF5.append_raw_data()
    HDF5.append_conc_data()
    # Include return for debugging
    return(HDF5)
# TODO - Include methods to import arrays.

if __name__ == '__main__':
    HDF5 = main()
