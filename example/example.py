from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger

import os

configuration_path='/example/config/config.ini'
# Load Configuration file
import configparser
config = configparser.ConfigParser()
config.read(check_create_folder(configuration_path))


# Setup logging
setup_logger(config['Paths']['log'],config['General']['logging_level'] )

from src.classes.nw_preprocessor import nw_preprocessor

# Set up preprocessor
preprocessor = nw_preprocessor(config)
# Set up logging
#preprocessor.setup_logger()



#preprocessor.preprocess_folders(overwrite=True,excludelist=['checked', 'Error'])
#preprocessor.preprocess_files(overwrite=True,excludelist=['checked', 'Error'])


from src.classes.bert_trainer import bert_trainer

trainer=bert_trainer(config)
#trainer.train_berts()


from src.classes.neo4jnw import neo4j_network
neograph = neo4j_network(config)


from src.classes.nw_processor import nw_processor
processor = nw_processor(neograph,config)



processor.run_all_queries(delete_incomplete=True, delete_all=True)
