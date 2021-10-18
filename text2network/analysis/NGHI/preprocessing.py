# %% Note on working directory
# This loads from text2network as if it were a normal python package (which it is)
# This means, the text2network folder must be visible
# So the best way is to set the working directory for python to
# D:\NLP\text2network


# %% Imports
import configparser

from text2network.preprocessing.nw_preprocessor import nw_preprocessor

from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = '/config/config.ini'
# Load Configuration file
config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'],config['General']['logging_level'], "preprocessing_and_training")


##################### Preprocessing


# Set up preprocessor
preprocessor = nw_preprocessor(config)

# Preprocess file
# We have two options: Folders and Files
preprocessor.preprocess_folders(config['Paths']['import_folder'],overwrite=True,excludelist=['checked', 'Error'])


# If Instead all files are in a single folder:
#preprocessor.preprocess_files(config['Paths']['import_folder'],excludelist=['acad', 'fic','spok','mag'])


##################### Training

# From now on, everything should run automatically

#trainer=bert_trainer(config)
#trainer.train_berts()
