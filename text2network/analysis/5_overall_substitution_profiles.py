
from itertools import product

import pandas as pd

from text2network.measures.proximity import get_top_100
from text2network.utils.file_helpers import check_create_folder, check_folder
from text2network.measures.measures import average_cluster_proximities
from text2network.utils.logging_helpers import setup_logger
import logging
import numpy as np
from text2network.functions.graph_clustering import consensus_louvain
from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.role_profiles import get_pos_profile, create_YOY_role_profile, extract_yoy_role_profiles, \
    get_clustered_role_profile

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset=None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "5_overall_substitution_profiles.py")

semantic_network = neo4j_network(config)


times = list(range(1980, 2021))
focal_token="leader"
sym=False
keep_top_k = 40
max_degree=200
rs=100
cutoff=0
level=10
depth=1
context_mode="bidirectional"
compositional=False
rev=False
ma=(2,2)

cent = get_top_100(semantic_network=semantic_network, focal_tokens=focal_token, times=times, symmetric=sym, compositional=compositional,
                   reverse=rev)
top100 = list(cent.manager[0:100].index)

filename = "".join(
    [config['Paths']['csv_outputs'], "/ClusteredRoles", str(focal_token), "__", str('top100'),"_t3", "_max_degree", str(max_degree), "_sym",
     str(sym),"_rev",
     str(rev),
     "_lev", str(level), "_ma", str(ma), "_cut",str(cutoff), "_keeptopk", str(keep_top_k), "_cm", str(context_mode),
      "_depth", str(depth), "_rs", str(rs)])
df=get_clustered_role_profile(semantic_network,focal_token=focal_token, cluster_nodes=top100, times=times, keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, weight_cutoff=cutoff, level=level,
                              depth=depth, context_mode=context_mode, filename=filename)

times = list(range(1980, 2021))

context_mode = "bidirectional"
filename = "".join(
    [config['Paths']['csv_outputs'], "/RoleCluster", str(focal_token), "_max_degree", str(max_degree), "_sym",
     str(sym),
     "_lev", str(level), "_ma", str(ma), "_cut", "_keeptopk", str(keep_top_k), "_cm", str(context_mode),
     str(cutoff), "_depth", str(depth), "_rs", str(rs)])
logging.info("YOY Profiles : {}".format(filename))
extract_yoy_role_profiles(semantic_network=semantic_network,df_cluster_ids=df_cluster_ids, df_clusters=df_clusters, df_nodes=df_nodes, times=times, focal_token=focal_token, keep_top_k=keep_top_k, cutoff=cutoff,
                              depth=depth, context_mode=context_mode, moving_average=ma, filename=filename)

