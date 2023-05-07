import logging

from text2network.utils.file_helpers import check_create_folder
import functools


def log(logging_level=None, other_loggers=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the decorator is applied to a class method
            if args and hasattr(args[0], "__class__"):
                instance = args[0]
                if hasattr(instance, "logging_level") and hasattr(
                    instance, "other_loggers"
                ):
                    logging_level_inner = instance.logging_level
                    other_loggers_inner = instance.other_loggers
                else:
                    logging_level_inner = logging_level
                    other_loggers_inner = other_loggers
            else:
                logging_level_inner = logging_level
                other_loggers_inner = other_loggers

            logging.basicConfig(level=other_loggers_inner)

            t2n_logger = logging.getLogger("t2n")
            t2n_logger.setLevel(logging_level_inner)

            # Call the original function and return the result
            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


def setup_logger(logging_path, logging_level, filename="log"):
    """
    Sets up logging formats and file etc.
    Returns
    -------
    None
    """
    logging_level = int(logging_level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
    )
    # Logging path
    logging_path = check_create_folder(logging_path)

    # Set up logging
    logging.info("Setting loggging level {}".format(logging_level))

    logging.getLogger("t2n").setLevel(logging_level)
    rootLogger = logging.getLogger("t2n")
    logFormatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    )
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(logging_path, "".join(["text2network_", filename]))
    )
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
