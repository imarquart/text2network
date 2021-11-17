from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.utils.load_bert import get_only_tokenizer
from text2network.utils.logging_helpers import setup_logger
from text2network.datasets.text_dataset import bert_dataset, bert_dataset_old, query_dataset

# Set a configuration path
configuration_path = 'config/2021/SenBert40.ini'
# Settings
years = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path, False))
config.read(check_create_folder(configuration_path, False))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "dataset_debug.py")


db=config['Paths']['database']
bert=config['Paths']['trained_berts']+"/2016"

path=check_folder(db)
bert=check_folder(bert)

tokenizer=get_only_tokenizer(bert)

dataset=query_dataset(data_path=path, tokenizer=tokenizer, fixed_seq_length=40, maxn=None, query="year==2016")

b_dataset=bert_dataset(tokenizer=tokenizer,database=path,where_string="year==2016",block_size=40)


for i in range (0,10):
    print(tokenizer.convert_ids_to_tokens(b_dataset[-i].tolist()))
    print("--------------------------------")