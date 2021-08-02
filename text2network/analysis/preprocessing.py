# %% Imports

# Load Configuration file
import configparser
import json

from src.classes.bert_trainer import bert_trainer
from src.classes.neo4jnw import neo4j_network
from src.classes.nw_preprocessor import nw_preprocessor
from src.classes.nw_processor import nw_processor

config = configparser.ConfigParser()
config.read('D:/NLP/COCA/cocaBERT/config/config.ini')
logging_level = config['General'].getint('logging_level')

neo_creds = (config['NeoConfig']["db_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))

# Set up preprocessor
preprocessor = nw_preprocessor(config)
# Set up logging
preprocessor.setup_logger()

# Preprocess file
#preprocessor.preprocess_folders(config['Paths']['import_folder'],overwrite=True,excludelist=['checked', 'Error'])
#preprocessor.preprocess_files(config['Paths']['import_folder'],excludelist=['acad', 'fic','spok','mag'])

trainer=bert_trainer(config)
trainer.train_berts()

neograph = neo4j_network(neo_creds)

processor = nw_processor(config['Paths']['trained_berts'], neograph,
                            config['Preprocessing'].getint('max_seq_length'), config['Processing'],
                            text_db=config['Paths']['database'],
                            split_hierarchy=json.loads(config.get('General', 'split_hierarchy')),
                            processing_cache=config['Paths']['processing_cache'],
                            logging_level=config['General'].getint('logging_level'))
processor.run_all_queries(clean_database=True)


