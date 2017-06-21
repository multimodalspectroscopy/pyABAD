import os


def base_dir_check(fpath, expected_dir, verbose=False):
    """
    Fucntio nto check that the current directory is the expected one.
    :param fpath: Actual file path to be checked.
    :param expected_dir: The expected directory.
    :param verbose: Switch for turning on/off verbosity (for logging).
    :return: Boolean true or false
    """

    BASEDIR = os.path.basename(fpath)
    if verbose:
        print("Expected Directory: %s\nActual Directory: %s\n" %
              (expected_dir, BASEDIR))
    return BASEDIR == expected_dir
