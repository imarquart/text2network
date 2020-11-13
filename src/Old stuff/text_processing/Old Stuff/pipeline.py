# %% Imports


import glob
import logging
import os
import time

import networkx as nx
import torch

from NLP.config.config import configuration
from NLP.src.novelty import entropy_network
from NLP.src.process_sentences_network import process_sentences_network
from NLP.src.reduce_network import reduce_network, moving_avg_networks, min_symmetric_network,save_merged_ego_graph
from NLP.src.run_bert import bert_args, run_bert
from NLP.src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from NLP.src.utils.hash_file import hash_file, check_step, complete_step
from NLP.utils.load_bert import get_bert_and_tokenizer

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
    nw_folder = ''.join([data_folder, cfg.nw_folder])
    sum_folder = ''.join([data_folder, cfg.sums_folder])
    np_folder = ''.join([data_folder, cfg.np_folder])
    ma_folder = ''.join([data_folder, cfg.ma_folder])
    entropy_folder = ''.join([data_folder, cfg.entropy_folder])
    sumsym_folder = ''.join([data_folder, cfg.sumsym_folder])
    text_file = ''.join([text_folder, '/', str(year), '.txt'])
    text_db = ''.join([text_folder, '/', str(year), '.h5'])
    plot_folder = ''.join([cfg.data_folder, '/plots'])

    if not os.path.exists(text_folder): os.mkdir(text_folder)
    if not os.path.exists(data_folder): os.mkdir(data_folder)
    if not os.path.exists(bert_folder): os.mkdir(bert_folder)
    if not os.path.exists(tensor_folder): os.mkdir(tensor_folder)
    if not os.path.exists(nw_folder): os.mkdir(nw_folder)
    if not os.path.exists(sum_folder): os.mkdir(sum_folder)
    if not os.path.exists(np_folder): os.mkdir(np_folder)
    if not os.path.exists(plot_folder): os.mkdir(plot_folder)
    if not os.path.exists(ma_folder): os.mkdir(ma_folder)
    if not os.path.exists(entropy_folder): os.mkdir(entropy_folder)
    if not os.path.exists(sumsym_folder): os.mkdir(sumsym_folder)

    logging.info("Copying separate text files into text folder.")
    read_files = glob.glob(''.join([input_folder, '/*.txt']))

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
        preprocess_files_HBR(input_folder, text_db, cfg.max_seq_length, cfg.char_mult, max_seq=cfg.max_seq)
        logging.disable(logging.NOTSET)
        logging.info("Pre-processing finished in %s seconds" % (time.time() - start_time))
        complete_step(text_folder, hash)

    # %% Train BERT
    logging.info("Training BERT")
    if (check_step(bert_folder, hash) and False) or year in range(1990,2002):
        logging.info("Found trained BERT. Skipping")
    else:

        start_time = time.time()
        # Create BERT args
        args = bert_args(text_file, bert_folder, cfg.do_train, cfg.model_dir, cfg.mlm_probability, cfg.max_seq_length,
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
    if (check_step(nw_folder, hash) and False) or year in range(1990,2002):
        logging.info("Processed results found. Skipping.")
    else:
        start_time = time.time()
        torch.cuda.empty_cache()
        logging.disable(cfg.subprocess_level)
        tokenizer, bert = get_bert_and_tokenizer(bert_folder, True)
        logging.disable(logging.NOTSET)
        DICT_SIZE = tokenizer.vocab_size
        # Process sentences in BERT and create the networks
        graph, context_graph, attention_graph = process_sentences_network(tokenizer, bert, text_db, cfg.max_seq_length,
                                                                          DICT_SIZE,
                                                                          cfg.batch_size,
                                                                          nr_workers=0,
                                                                          cutoff_percent=cfg.cutoff_percent,
                                                                          max_degree=cfg.max_degree)

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
        logging.info("Attempting to save gefx")
        if not os.path.exists(graph_path) or True:
            logging.info("Saving to %s" %graph_path)
            nx.write_gexf(graph, graph_path)
        else:
            logging.info("Full graph %s exists, please confirm overwrite manually!" % graph_path)

        del graph
        graph_path = os.path.join(nw_folder, "".join(['Cgraph.gexf']))
        if not os.path.exists(graph_path) or True:
            logging.info("Saving to %s" %graph_path)
            nx.write_gexf(context_graph, graph_path)
        else:
            logging.info("Full graph %s exists, please confirm overwrite manually!" % graph_path)

        del context_graph
        graph_path = os.path.join(nw_folder, "".join(['Agraph.gexf']))
        if not os.path.exists(graph_path):
            nx.write_gexf(attention_graph, graph_path)
        else:
            logging.info("Full graph %s exists, please confirm overwrite manually!" % graph_path)

        del attention_graph

        logging.info("Network creation finished in %s seconds" % (time.time() - start_time))
        complete_step(nw_folder, hash)

    # %% Sum Networks

    logging.info("Summing graphs.")
    if (check_step(sum_folder, hash) and False) or year in range(1990,2002):
        logging.info("Summed graphs found. Skipping.")
    else:
        start_time = time.time()
        for big_graph in ['Rgraph','Cgraph']:
            logging.info("Summing %s" % big_graph)
            graph_path = os.path.join(nw_folder, "".join([big_graph,'.gexf']))
            if os.path.exists(graph_path):
                save_folder = os.path.join(sum_folder, "".join([big_graph,'-Sum.gexf']))
                _ = reduce_network(graph_path, cfg, reverse=False, method="sum", save_folder=save_folder)
                save_folder = os.path.join(sum_folder, "".join([big_graph,'-Sum-Rev.gexf']))
                _ = reduce_network(graph_path, cfg, reverse=True, method="sum", save_folder=save_folder)
            else:
                logging.info("Note that path %s was not found for summation" % graph_path)

        logging.info("Graph summation finished in %s seconds" % (time.time() - start_time))
        complete_step(sum_folder, hash)

    # %% Symmetrize network
    if (check_step(sumsym_folder,hash) and False) or year in range(1990,2001):
        logging.info("Symmetrized graphs found. Skipping.")
    else:
        start_time = time.time()
        for network_type in ["Rgraph-Sum-Rev","Cgraph-Sum-Rev"]:
            logging.info("Symmetry processing on %s" % network_type)
            min_symmetric_network(year, cfg, cfg.sums_folder, cfg.sumsym_folder, network_type, method="min-sym-avg")
            min_symmetric_network(year, cfg, cfg.sums_folder, cfg.sumsym_folder, network_type, method="min")
        logging.info("Symmetry processing finished in %s seconds" % (time.time() - start_time))
        complete_step(sumsym_folder, hash)

    # %% No Plural Networks

    logging.info("Summing graphs without plurals.")
    if (check_step(np_folder,hash) and False) or year in range(1990,2001):
        logging.info("Summed graphs found. Skipping.")
    else:
        start_time = time.time()
        for big_graph in ['Rgraph','Cgraph']:
            graph_path = os.path.join(nw_folder, "".join([big_graph,'.gexf']))
            logging.info("Summing %s" % graph_path)

            if os.path.exists(graph_path):
                save_folder = os.path.join(np_folder, "".join([big_graph,'-Sum-Rev-NP.gexf']))
                _ = reduce_network(graph_path, cfg, reverse=True, method="sum", save_folder=save_folder,plural_elim=True)
            else:
                logging.info("Note that path %s was not found for summation" % graph_path)

        logging.info("Graph plural elimination finished in %s seconds" % (time.time() - start_time))
        complete_step(np_folder, hash)


    # %% Entropy networks

    if (check_step(entropy_folder,hash) and False) or year in range(1990,2001):
        logging.info("Entropy graphs found. Skipping.")
    else:
        start_time = time.time()
        for network_type in ["Rgraph"]:
            for focal_token in cfg.focal_nodes:
                logging.info("Entropy processing on %s" % focal_token)
                save_folder=''.join([entropy_folder,'/entropy_',focal_token,'_',network_type,'.gexf'])
                entropy_network(focal_token, year, cfg, network_type, save_folder)
        logging.info("Entropy processing finished in %s seconds" % (time.time() - start_time))
        complete_step(entropy_folder, hash)

    logging.info("---------- Finished year %i ----------" % year)

# %% Create Mergend networks
olds_folder="/networks/sums"
mfolder=''.join([cfg.merged_folder])
if check_step(mfolder, "1") and False:
    logging.info("Merged graphs found. Skipping.")
else:
    start_time = time.time()
    for ego_radius in [1,2]:
        for network_type in ["Rgraph-Sum-Rev","Cgraph-Sum-Rev"]:
            for focal_token in cfg.focal_nodes:
                logging.info("Merging %s on %s" % (focal_token, network_type))
                save_merged_ego_graph(range(1990,2019), focal_token, cfg, cfg.merged_folder, olds_folder, ego_radius=ego_radius, links="both",
                                      network_type=network_type)
    logging.info("Merged graphs finished in %s seconds" % (time.time() - start_time))
    #complete_step(mfolder, "1")

# %% MA Networks

if check_step(ma_folder, "1") and False:
    logging.info("Moving average graphs found. Skipping.")
else:
    start_time = time.time()
    for network_type in ["Rgraph-Sum-Rev","Cgraph-Sum-Rev"]:
        # for network_type in ["Rgraph-Sum-Rev", "Rgraph-Sum", "Cgraph-Sum", "Cgraph-Sum-Rev", "Agraph-Sum", "Agraph-Sum-Rev"]:
        logging.info("MA processing on %s" % network_type)
        moving_avg_networks(years, cfg, cfg.ma_order, network_type, cfg.average_links,load_all=True)
    logging.info("Graph MA processing finished in %s seconds" % (time.time() - start_time))

    complete_step(ma_folder, "1")

    # %% TODO: Symmetrization
