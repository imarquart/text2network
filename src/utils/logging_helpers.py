import logging
from src.functions.file_helpers import check_create_folder

def setup_logger(logging_path, logging_level):
    """
    Sets up logging formats and file etc.
    Returns
    -------
    None
    """
    # Logging path
    logging_path=check_create_folder(logging_path)
    logging_level=int(logging_level)
    # Set up logging

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging_level)

    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(logging_path, "preprocessing"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)