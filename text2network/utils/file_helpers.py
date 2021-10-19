import logging
from os import mkdir, getcwd
from os.path import exists, normpath,join
from pathlib import Path

def check_folder(folder):
    return check_create_folder(folder,False)

def check_create_folder(folder,create_folder=True):
    """
    Checks and creates folder if necessary
    If path relative, checks also from current working directory
    Parameters
    ----------
    create_folder
    folder: String
        Folder

    Returns
    -------
    folder directory on success

    """

    folder = Path(folder)
    # Make absolute
    if not folder.is_absolute():
        folder = Path.cwd() / folder


    # check if file
    if not folder.suffix == '':
        filename=folder.stem+folder.suffix
        db_folder=str(folder.parents[0])
    else:
        filename=""
        db_folder=str(folder)

    if not exists(db_folder):
        if create_folder:
            try:
                logging.info("Folder {} does not exist. Creating folder.".format(db_folder))
                mkdir(db_folder)
            except:
                msg="Could not create folder {}".format(db_folder)
                logging.error(msg)
                raise AttributeError(msg)
            return normpath(db_folder+"/"+filename)
        else:
                msg="Folder {} does not exist".format(db_folder)
                logging.error(msg)
                raise AttributeError(msg)
    else:
        return normpath(db_folder+"/"+filename)
