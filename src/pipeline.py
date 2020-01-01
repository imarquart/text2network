#%% Imports


import glob
import os, time, sys
import logging
from shutil import copyfile
from sys import exit
from NLP.src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from NLP.config.config import configuration
from NLP.src.process_sentences_network import process_sentences_network
from NLP.utils.load_bert import get_bert_and_tokenizer
from NLP.utils.hash_file import hash_file,check_step,complete_step
from NLP.src.run_bert import bert_args, run_bert
import torch
import networkx as nx


#%% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg=configuration()



#%% Main Loop:
years=range(1990,2019)

for year in years:
    logging.info("---------- Starting year %i ----------" % year)
    input_folder='D:/NLP/BERT-NLP/NLP/data/HBR/articles/'
    input_folder=''.join([input_folder,str(year)])

    #%% Create folder structure
    logging.info("Setting up folder structure")
    text_folder=''.join([cfg.text_folder,'/',str(year)])
    data_folder=''.join([cfg.data_folder,'/',str(year)])
    bert_folder=''.join([data_folder,'/bert'])
    tensor_folder=''.join([data_folder,'/tensors'])
    nw_folder=''.join([data_folder,'/networks'])
    text_file=''.join([text_folder,'/',str(year),'.txt'])
    text_db=''.join([text_folder,'/',str(year),'.h5'])

    if not os.path.exists(text_folder): os.mkdir(text_folder)
    if not os.path.exists(data_folder): os.mkdir(data_folder)
    if not os.path.exists(bert_folder): os.mkdir(bert_folder)
    if not os.path.exists(tensor_folder): os.mkdir(tensor_folder)
    if not os.path.exists(nw_folder): os.mkdir(nw_folder)

    logging.info("Copying separate text files into text folder.")
    read_files = glob.glob(''.join([input_folder,'/*.txt']))

    with open(text_file, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())



    #%% Get file hash
    hash=hash_file(text_file,hash_factory="md5")
    logging.info("File Hash is %s" % hash)

    #%% Preprocess files
    logging.info("Pre-processing text file")
    if check_step(text_folder, hash):
        logging.info("Pre-processing already completed, skipping")
    else:
        start_time = time.time()
        logging.disable(cfg.subprocess_level)
        preprocess_files_HBR(input_folder,text_db,cfg.max_seq_length,cfg.char_mult,max_seq=cfg.max_seq)
        logging.disable(logging.NOTSET)
        logging.info("Pre-processing finished in %s seconds" % (time.time() - start_time))
        complete_step(text_folder,hash)

    #%% Train BERT
    logging.info("Training BERT")
    if check_step(bert_folder, hash):
        logging.info("Found trained BERT. Skipping")
    else:

        start_time = time.time()
        # Create BERT args
        args = bert_args(text_file, bert_folder, cfg.do_train, cfg.model_dir, cfg.mlm_probability, cfg.max_seq_length, cfg.gpu_batch, cfg.epochs,
                         cfg.warmup_steps)
        torch.cuda.empty_cache()
        logging.disable(cfg.subprocess_level)
        results=run_bert(args)
        logging.disable(logging.NOTSET)
        logging.info("BERT results %s" % results)
        logging.info("BERT training finished in %s seconds" % (time.time() - start_time))
        complete_step(bert_folder,hash)


    #%% Process files, create networks

    logging.info("Processing text to create networks.")
    if check_step(nw_folder, hash):
        logging.info("Processed results found. Skipping.")
    else:
        start_time = time.time()
        torch.cuda.empty_cache()
        logging.disable(cfg.subprocess_level)
        tokenizer, bert = get_bert_and_tokenizer(bert_folder, True)
        logging.disable(logging.NOTSET)
        DICT_SIZE = tokenizer.vocab_size
        # Process sentences in BERT and create the networks
        graph, context_graph, attention_graph = process_sentences_network(tokenizer, bert, text_db, cfg.max_seq_length, DICT_SIZE,
                                                                      cfg.batch_size,
                                                                      nr_workers=0,
                                                                      cutoff_percent=cfg.cutoff_percent)

        logging.info("Relabeling and saving graphs")
        token_map = {v: k for k, v in tokenizer.vocab.items()}

        # Label nodes by token
        graph = nx.relabel_nodes(graph, token_map)
        context_graph = nx.relabel_nodes(context_graph, token_map)
        attention_graph = nx.relabel_nodes(attention_graph, token_map)

        # Take edge subgraph: Delete non-needed nodes
        graph = graph.edge_subgraph(graph.edges)
        context_graph = context_graph.edge_subgraph(context_graph.edges)
        attention_graph = attention_graph.edge_subgraph(attention_graph.edges)

        # Save Graphs
        graph_path = os.path.join(nw_folder, "".join(['Rgraph.gexf']))
        nx.write_gexf(graph, graph_path)
        graph_path = os.path.join(nw_folder, "".join(['Cgraph.gexf']))
        nx.write_gexf(context_graph, graph_path)
        graph_path = os.path.join(nw_folder, "".join(['Agraph.gexf']))
        nx.write_gexf(attention_graph, graph_path)

        logging.info("Network creation finished in %s seconds" % (time.time() - start_time))
        complete_step(nw_folder,hash)
        logging.info("---------- Finished year %i ----------" % year)
