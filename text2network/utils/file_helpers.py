import logging
from os import makedirs, mkdir
from os.path import exists, normpath
from pathlib import Path


def check_folder(folder):
    return check_create_folder(folder, False)


import logging
from os import makedirs, mkdir
from os.path import exists, normpath
from pathlib import Path


def check_create_folder(folder, create_folder=True):
    """
    Checks and creates folder if necessary
    If path relative, checks also from current working directory
    Takes into account if a file is passed, or a folder

    Parameters
    ----------
    create_folder
    folder: String
        Folder

    Returns
    -------
    folder directory on success

    """
    folder_path = Path(folder).resolve()

    if folder_path.suffix != "":
        filename = folder_path.name
        parent_folder = folder_path.parent
    else:
        filename = ""
        parent_folder = folder_path

    if not parent_folder.exists():
        if create_folder:
            try:
                logging.info(f"Folder {parent_folder} does not exist. Creating folder.")
                parent_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                msg = f"Could not create folder {parent_folder}"
                logging.error(msg)
                raise AttributeError(msg) from e
            return str(folder_path)
        else:
            msg = f"Folder {parent_folder} does not exist"
            logging.error(msg)
            raise AttributeError(msg)
    else:
        return str(folder_path)
