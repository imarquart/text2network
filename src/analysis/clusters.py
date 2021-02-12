from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.functions.node_measures import proximity, centrality
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "clustering")

# First, create an empty network
semantic_network = neo4j_network(config)

# years=np.array(semantic_network.get_times_list())
# years=-np.sort(-years)
logging.info("------------------------------------------------")
years = range(1980, 2020)
focal_words = ["leader", "manager"]
focal_words2 = ["leader"]
alter_subset = ["boss", "coach", "consultant", "expert", "mentor", "superior"]

levels = [8]
cutoffs = [0]
depths = [0]

for level, cutoff, depth in zip(levels, cutoffs, depths):
    logging.info("------------------------------------------------")
    filename = "".join(
        [config['Paths']['csv_outputs'], "/egocluster_lev", str(level), "_cut", str(cutoff), "_depth", str(depth),
         ".xlsx"])
    filename_csv = "".join(
        [config['Paths']['csv_outputs'], "/egocluster_lev", str(level), "_cut", str(cutoff), "_depth", str(depth),
         ".csv"])
    logging.info("Network clustering: {}".format(filename))
    filename = check_create_folder(filename)
    filename_csv = check_create_folder(filename_csv)
    del semantic_network
    semantic_network = neo4j_network(config, connection_type="bolt")
    start_time = time.time()
    dataframe_list = []
    if depth > 0:
        clusters = semantic_network.cluster(ego_nw_tokens="leader", depth=depth, levels=level, weight_cutoff=cutoff,
                                            to_measure=[proximity])
    else:
        clusters = semantic_network.cluster(levels=level, weight_cutoff=cutoff, to_measure=[proximity])
    for cl in clusters:
        if cl['level'] == 0:
            rev_proxim = semantic_network.pd_format(cl['measures'])[0].loc[:, 'leader']
            proxim = semantic_network.pd_format(cl['measures'])[0].loc['leader', :]
        if len(cl['graph'].nodes) > 2 and cl['level'] > 0:
            nodes = semantic_network.ensure_tokens(list(cl['graph'].nodes))
            proximate_nodes = proxim.reindex(nodes, fill_value=0)
            proximate_nodes = proximate_nodes[proximate_nodes > 0]
            mean_cluster_prox = np.mean(proximate_nodes)
            top_node = proximate_nodes.idxmax()
            if len(proximate_nodes) > 0:
                logging.info("Name: {}, Level: {}, Parent: {}".format(cl['name'], cl['level'], cl['parent']))
                logging.info("Nodes: {}".format(nodes))
                logging.info(proximate_nodes)
            for node in nodes:
                if proxim[node] > 0:
                    node_prox = proxim.reindex([node], fill_value=0)[0]
                    node_rev_prox = rev_proxim.reindex([node], fill_value=0)[0]
                    delta_prox = node_prox - node_rev_prox
                    df_dict = {'Token': node, 'Level': cl['level'], 'Clustername': cl['name'], 'Prom_Node': top_node,
                               'Parent': cl['parent'], 'Cluster_Avg_Prox': mean_cluster_prox, 'Proximity': node_prox,
                               'Rev_Proximity': node_rev_prox, 'Delta_Proximity': delta_prox}
                    dataframe_list.append(df_dict.copy())

    df = pd.DataFrame(dataframe_list)
    df.to_excel(filename)
    df.to_csv(filename_csv)
    del clusters
