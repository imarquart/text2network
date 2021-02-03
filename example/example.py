from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger
from src.functions.format import pd_format

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
#preprocessor = nw_preprocessor(config)
# Set up logging
#preprocessor.setup_logger()



#preprocessor.preprocess_folders(overwrite=True,excludelist=['checked', 'Error'])
#preprocessor.preprocess_files(overwrite=True,excludelist=['checked', 'Error'])


from src.classes.bert_trainer import bert_trainer

#trainer=bert_trainer(config)
#trainer.train_berts()


from src.classes.neo4jnw import neo4j_network
semantic_network = neo4j_network(config)


from src.classes.nw_processor import nw_processor
#processor = nw_processor(semantic_network,config)
#processor.run_all_queries(delete_incomplete=True, delete_all=True)


semantic_network = neo4j_network(config)
print(semantic_network.pd_format(semantic_network.proximities(['nation'])))
semantic_network.set_norm_ties()
semantic_network.set_norm_ties()

semantic_network.get_times_list()


semantic_network.condition(years=[1789])
print(semantic_network.pd_format(semantic_network.proximities(['nation']))[0].iloc[:5,:])
semantic_network.decondition()
print(semantic_network.pd_format(semantic_network.proximities(['nation'],years=[1789]))[0].iloc[:5,:])
semantic_network.condition(years=[2009])
print(semantic_network.pd_format(semantic_network.proximities(['nation']))[0].iloc[:5,:])
semantic_network.decondition()
print(semantic_network.pd_format(semantic_network.proximities(['nation'],years=[2009]))[0].iloc[:5,:])



print(semantic_network.pd_format(semantic_network.proximities(['president'])))
print(semantic_network.pd_format(semantic_network.centralities()))
semantic_network.to_backout()
