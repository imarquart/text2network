# %% Imports

# Trying to use only one GPU, the one less used??
import os
def find_gpus(nums=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freeMemory_pair[:nums] ]
    usingGPUs =  ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs
#os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=1)

import argparse
import configparser
import logging
import sys
import traceback
import torch
from text2network.training.bert_trainer import bert_trainer
import gc
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

import nltk
nltk.download('all')

def run_training(args):
    # Set a configuration path
    configuration_path = args.config
    # Load Configuration file
    config = configparser.ConfigParser()
    print("Loading config in {}".format(configuration_path))
    try:
        config.read(check_create_folder(configuration_path))
    except:
        logging.error("Could not read config.")
    # Setup logging
    logger = setup_logger(config['Paths']['log'], config['General']['logging_level'], "training.py")

    ##################### Training
    trainer=bert_trainer(config)
    return trainer.train_berts()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text files.')
    parser.add_argument('--config', metavar='path', required=True,
                        help='the path to the configuration file')
    args = parser.parse_args()
    
    success=False
    retries=0
    while success is not True:
        try:
            sucvar=run_training(args)
        except:
            etype, value, _ = sys.exc_info()
            logging.error("Error in train_berts(): {}".format(value))
            logging.error("Traceback: {}".format(traceback.format_exc()))

            # Here we can try to release CUDA memory

            gc.collect()
            print(torch.cuda.is_available())
            torch.cuda.empty_cache()
            logging.info("{}: trying to continue".format(value))

            retries+=1
            #raise
            continue
        if sucvar == 0:
            logging.info("Successfully trained BERTs, canceling")
            success=True
        if retries > 5:
            logging.error("No success after five retries, canceling")
            success=True


