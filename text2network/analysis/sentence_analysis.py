from itertools import product

import pandas as pd
import tables
from text2network.functions.file_helpers import check_create_folder
from text2network.measures.measures import average_cluster_proximities, extract_all_clusters, proximities
from text2network.utils.load_bert import get_bert_and_tokenizer
from text2network.utils.logging_helpers import setup_logger
from text2network.datasets.text_dataset import query_dataset

import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain, louvain_cluster
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "td_idf.py")


path="E:\\NLPHBR\\database\\db.h5"
bert="E:\\NLPHBR\\trained_berts\\2020"
focal_tokens=["aand"]
#focal_tokens=["azerbaijani","cleric", "patriarch","sol", "sebastien"]
for focal_token in focal_tokens:


    table = tables.open_file(path, mode="r")
    data = table.root.textdata.table

    items = data.read()[:]


    founder_data=[]
    for row in data:
        if focal_token in row['text'].decode("utf-8").lower():
            #print(row['text'])
            items_text = row['text'].decode("utf-8")
            items_year = row['year']
            items_seqid = row['seq_id']  # ?
            items_runindex = row['run_index']
            items_p1 = row['p1']
            items_p2 = row['p2']
            items_p3 = row['p3']
            items_p4 = row['p4']
            row_dict={"year": items_year, "seq_id": items_seqid, "run_index": items_runindex, "text": items_text, "p1": items_p1,
         "p2": items_p2,
         "p3": items_p3, "p4": items_p4}
            founder_data.append(row_dict)


    data = pd.DataFrame(founder_data)

    filename = "".join(
        [config['Paths']['csv_outputs'], "/sentences_", str(focal_token), ".xlsx"])

    filename = check_create_folder(filename)
    data.to_excel(filename, merge_cells=False)

    table.close()