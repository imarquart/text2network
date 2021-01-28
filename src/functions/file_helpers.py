from os import listdir, mkdir
from os.path import isfile, join, abspath, exists, dirname, basename, normpath

import logging

def check_create_folder(folder):
    """
    Checks and creates folder if necessary
    Parameters
    ----------
    folder: String
        Folder

    Returns
    -------
    Success or Failure

    """

    db_folder = dirname(folder)
    if not exists(db_folder):
        try:
            logging.info("Folder {} does not exist. Creating folder.".format(db_folder))
            mkdir(db_folder)
        except:
            msg="Could not create folder {}".format(db_folder)
            logging.error(msg)
            raise AttributeError(msg)

    return True