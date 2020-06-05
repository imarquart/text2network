# %% Imports

import glob
import logging
import os
import time

import torch
from src.neo4j_network import neo4j_network

from config.config import configuration
from src.text_processing.neo4j.process_sentences_neo4j import process_sentences_neo4j
from src.text_processing.run_bert import bert_args, run_bert
from src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from src.text_processing.preprocess_files import pre_process_sentences_COCA
from src.utils.hash_file import hash_file, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer

# %% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg = configuration()
# Init Network
q_size = 150
t_size = 1
maxn = 15000
par = "nonpar"
db_uri = "http://localhost:7474"
db_pwd = ('neo4j', 'nlp')
neo_creds = (db_uri, db_pwd)


dataset="COCA"

if __name__ == "__main__":
    # %% Main Loop:
    years = range(1990, 2013)
    results = []
    for year in years:
        logging.info("---------- Starting year %i ----------" % year)
        if dataset=="COCA":
            input_folder = 'D:/NLP/BERT-NLP/NLP/data/COCA/'
        else:
            input_folder = 'D:/NLP/BERT-NLP/NLP/data/HBR/'
            input_folder = ''.join([input_folder, str(year)])

        # %% Create folder structure
        logging.info("Setting up folder structure")
        text_folder = ''.join([cfg.text_folder, '/', str(year)])
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        bert_folder = ''.join([data_folder, '/bert'])
        if dataset=="COCA":
            text_folder = ''.join([text_folder, "/HBR"])
            data_folder = ''.join([data_folder, "/HBR"])
            text_file = ''.join([text_folder, '/w_news_', str(year), '.txt'])
        else:
            text_file = ''.join([text_folder, '/', str(year), '.txt'])
        text_db = ''.join([text_folder, '/', str(year), '.h5'])
        plot_folder = ''.join([cfg.data_folder, '/plots'])

        if not os.path.exists(text_folder): os.mkdir(text_folder)
        if not os.path.exists(data_folder): os.mkdir(data_folder)
        if not os.path.exists(bert_folder): os.mkdir(bert_folder)
        if not os.path.exists(plot_folder): os.mkdir(plot_folder)

        # Process text files
        if not dataset=="COCA":
            logging.info("Copying separate text files into text folder.")
            read_files = glob.glob(''.join([input_folder, '/*.txt']))
        else:
            logging.info("Copying COCA text file into text folder.")
            read_files = glob.glob(''.join([input_folder,'/w_news_', str(year), '.txt']))

        with open(text_file, "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())

        # %% Get file hash
        hash = hash_file(text_file, hash_factory="md5")
        logging.info("File Hash is %s" % hash)

        # %% Preprocess files
        logging.info("Pre-processing text file")
        if check_step(text_folder, hash):
            logging.info("Pre-processing already completed, skipping")
        else:
            start_time = time.time()
            logging.disable(cfg.subprocess_level)
            if dataset=="COCA":
                pre_process_sentences_COCA([text_file], text_db, cfg.max_seq_length, cfg.char_mult, max_seq=cfg.max_seq)
            else:
                preprocess_files_HBR(input_folder, text_db, cfg.max_seq_length, cfg.char_mult, max_seq=cfg.max_seq)
            logging.disable(logging.NOTSET)
            logging.info("Pre-processing finished in %s seconds" % (time.time() - start_time))
            complete_step(text_folder, hash)

        # %% Train BERT
        logging.info("Training BERT")
        if (check_step(bert_folder, hash)) or dataset=="COCA":
            logging.info("Found trained BERT. Skipping")
        else:

            start_time = time.time()
            # Create BERT args
            args = bert_args(text_file, bert_folder, cfg.do_train, cfg.model_dir, cfg.mlm_probability,
                             cfg.max_seq_length,
                             cfg.loss_limit, cfg.gpu_batch, cfg.epochs,
                             cfg.warmup_steps)
            torch.cuda.empty_cache()
            logging.disable(cfg.subprocess_level)
            results = run_bert(args)
            logging.disable(logging.NOTSET)
            logging.info("BERT results %s" % results)
            logging.info("BERT training finished in %s seconds" % (time.time() - start_time))
            complete_step(bert_folder, hash)

        # %% Process files, create networks

        logging.info("Processing text to create networks.")
        if (False):
            logging.info("Processed results found. Skipping.")
        else:
            q_size = 150
            start_time = time.time()
            torch.cuda.empty_cache()
            logging.disable(cfg.subprocess_level)
            tokenizer, bert = get_bert_and_tokenizer(bert_folder, True)
            logging.disable(logging.NOTSET)
            DICT_SIZE = tokenizer.vocab_size
            year_var = int(''.join([str(year), "0101"]))

            # Re-setup graph
            neograph = neo4j_network(neo_creds, queue_size=q_size, tie_query_limit=t_size)

            # DEBUG
            nr_nodes = neograph.connector.run("MATCH (n) RETURN count(n) AS nodes")[0]['nodes']
            nr_ties = neograph.connector.run("MATCH ()-->() RETURN count(*) AS ties")[0]['ties']
            logging.info("Before cleaning: Network has %i nodes and %i ties" % (nr_nodes, nr_ties))

            # Delete previous year entries
            query = ''.join(
                ["MATCH (p:edge {time:", str(year_var), "}) WHERE p.time=", str(year_var), " DETACH DELETE p"])
            neograph.connector.run(query)

            # DEBUG
            nr_nodes = neograph.connector.run("MATCH (n) RETURN count(n) AS nodes")[0]['nodes']
            nr_ties = neograph.connector.run("MATCH ()-->() RETURN count(*) AS ties")[0]['ties']
            logging.info("After cleaning: Network has %i nodes and %i ties" % (nr_nodes, nr_ties))

            # Setup network
            tokens = list(tokenizer.vocab.keys())
            tokens = [x.translate(x.maketrans({"\"": '#e1#', "'": '#e2#', "\\": '#e3#'})) for x in tokens]
            neograph.setup_neo_db(tokens, list(tokenizer.vocab.values()))
            start_time = time.time()

            # Process sentences in BERT and create the networks
            process_sentences_neo4j(tokenizer, bert, text_db, neograph, year_var, cfg.max_seq_length,
                                        DICT_SIZE,
                                        cfg.batch_size, maxn=maxn,
                                        nr_workers=0,
                                        cutoff_percent=80,
                                        max_degree=50)
            logging.info("Par %s Network creation finished in %s seconds for q_size %i and t_size %i" % (par,
                                                                                                         time.time() - start_time,
                                                                                                         q_size,
                                                                                                         t_size))
            del tokenizer, bert
            torch.cuda.empty_cache()
            # Check results
            nr_nodes = neograph.connector.run("MATCH (n) RETURN count(n) AS nodes")[0]['nodes']
            nr_ties = neograph.connector.run("MATCH ()-->() RETURN count(*) AS ties")[0]['ties']
            logging.info("After execution: Network has %i nodes and %i ties" % (nr_nodes, nr_ties))
            results.append(((par, year_var, q_size, maxn, time.time() - start_time, nr_nodes, nr_ties)))
            # print(*results, sep="\n")
            del neograph

    print(*results, sep="\n")
