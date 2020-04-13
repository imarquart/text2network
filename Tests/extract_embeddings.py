

import glob
import logging
import os
import time
import pickle
import torch

from NLP.config.config import configuration
from NLP.src.process_embeddings import process_embeddings
from NLP.src.utils.hash_file import hash_file, check_step, complete_step

# %% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg = configuration()

# %% Main Loop:
years = cfg.years
for year in years:
    logging.info("---------- Starting year %i ----------" % year)
    input_folder = 'D:/NLP/BERT-NLP/NLP/data/HBR/articles/'
    input_folder = ''.join([input_folder, str(year)])

    # %% Create folder structure
    logging.info("Setting up folder structure")
    text_folder = ''.join([cfg.text_folder, '/', str(year)])
    data_folder = ''.join([cfg.data_folder, '/', str(year)])
    bert_folder = ''.join([data_folder, '/bert'])
    tensor_folder = ''.join([data_folder, '/tensors'])
    embed_folder = ''.join([data_folder, cfg.embed_folder])
    text_file = ''.join([text_folder, '/', str(year), '.txt'])
    text_db = ''.join([text_folder, '/', str(year), '.h5'])

    if not os.path.exists(text_folder): os.mkdir(text_folder)
    if not os.path.exists(data_folder): os.mkdir(data_folder)
    if not os.path.exists(bert_folder): os.mkdir(bert_folder)
    if not os.path.exists(tensor_folder): os.mkdir(tensor_folder)
    if not os.path.exists(embed_folder): os.mkdir(embed_folder)

    logging.info("Copying separate text files into text folder.")
    read_files = glob.glob(''.join([input_folder, '/*.txt']))

    with open(text_file, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

    # %% Get file hash
    hash = hash_file(text_file, hash_factory="md5")
    logging.info("File Hash is %s" % hash)

    # %% Process files, create networks

    logging.info("Processing text to create embeddings.")
    if check_step(embed_folder, hash):
        logging.info("Embeddings results found. Skipping.")
    else:
        start_time = time.time()
        torch.cuda.empty_cache()
        logging.disable(cfg.subprocess_level)
        logging.disable(logging.NOTSET)
        interest_set=['leadership']
        # Process sentences in BERT and create the networks
        pickle_dict = process_embeddings(bert_folder,text_db,interest_set, cfg.max_seq_length,cfg.batch_size)

        pickel_path = os.path.join(embed_folder, "".join([interest_set[0],'_embeddings.pcl']))
        logging.info("Attempting to save Pickle")
        if not os.path.exists(pickel_path):
            logging.info("Saving to %s" % pickel_path)
            outfile = open(pickel_path, 'wb')
            pickle.dump(pickle_dict, outfile)
            outfile.close()
        else:
            logging.info("Embeddings exist, please confirm overwrite! %s" % pickel_path)

        logging.info("Network creation finished in %s seconds" % (time.time() - start_time))
        complete_step(embed_folder, hash)

