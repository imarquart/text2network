import logging
from itertools import product

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain, infomap_cluster
from text2network.measures.context_profiles import context_per_pos, context_cluster_per_pos, context_cluster_all_pos
from text2network.measures.proximity import get_top_100
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "5_overall_substitution_profiles.py")

semantic_network = neo4j_network(config)


times = list(range(1980, 2021))
focal_token = "leader"
sym_list = [False]
keep_top_k_list = [100]
max_degree_list = [100]
rs_list = [100]
cutoff_list = [0.2, 0.1]
level_list = [6]
depth_list = [0, 1]
context_mode_list = ["bidirectional", "substitution", "occurring"]
rev_list = [False]
algo_list = [consensus_louvain, infomap_cluster]
ma_list = [(2, 2)]
keep_only_tokens_list = [True]
contextual_relations_list = [False]
pos_list = ["NOUN", "ADJ", "VERB"]
tfidf_list = [["rel_weight", "pmi_weight","weight","cond_entropy_weight" ]]
keep_top_k_list = [30,50,100,200,500]
max_degree_list = [50,100]
level_list = [10]
keep_only_tokens_list = [True]
contextual_relations_list = [True,False]

paraml1_list = product(cutoff_list,context_mode_list,tfidf_list)
param_list = product(rs_list,  level_list, depth_list,  max_degree_list, sym_list,
                     keep_top_k_list, ma_list, algo_list, contextual_relations_list, keep_only_tokens_list)


for cutoff, context_mode, tfidf in paraml1_list:
    logging.info("Extracting context tokens for {}, {}, {}".format(cutoff, context_mode, tfidf))
    context_dict = context_per_pos(snw=semantic_network, focal_substitutes=focal_token,
                                   times=times,
                                   weight_cutoff=cutoff,
                                   keep_top_k=None, context_mode=context_mode, pos_list=pos_list, tfidf=tfidf)

    for rs,  level, depth,  max_degree, sym, keep_top_k, ma, algo, contextual_relations, keep_only_tokens in param_list:
        filename = "".join(
            [config['Paths']['csv_outputs'], "/AllPosOvrlCluster", str(focal_token), "_max_degree", str(max_degree),
             "_algo", str(algo.__name__), "_depth", str(depth), "_conRel", str(contextual_relations),
             "_depth", str(depth), "_keeptopk", str(keep_top_k), "_cm", str(context_mode), "_keeponlyt_", str(depth == 0),
             str(cutoff), "_tfidf", str(tfidf is not None), "_lev", str(level),
             str(sym), "_rs", str(rs)])
        logging.info("Overall Profiles  ALL POS: {}".format(filename))
        context_cluster_all_pos(semantic_network, focal_substitutes=focal_token, times=times, keep_top_k=keep_top_k,
                                max_degree=max_degree, sym=sym, weight_cutoff=cutoff, level=level,
                                pos_list=pos_list, context_dict =context_dict,
                                depth=depth, context_mode=context_mode, algorithm=algo,
                                contextual_relations=contextual_relations, tfidf=tfidf, filename=filename)



times = list(range(1980, 2021))
focal_token = "leader"
sym_list = [False]
keep_top_k_list = [100]
max_degree_list = [100]
rs_list = [100]
cutoff_list = [0.2, 0.1]
level_list = [6]
depth_list = [0, 1]
context_mode_list = ["bidirectional", "substitution", "occurring"]
rev_list = [False]
algo_list = [consensus_louvain, infomap_cluster]
ma_list = [(2, 2)]
keep_only_tokens_list = [True]
contextual_relations_list = [False]
pos_list = ["NOUN", "ADJ", "VERB"]
tfidf_list = [["rel_weight", "pmi_weight","weight","cond_entropy_weight" ]]

paraml1_list = product(cutoff_list,context_mode_list,tfidf_list)
param_list = product(rs_list,  level_list, depth_list,  max_degree_list, sym_list,
                     keep_top_k_list, ma_list, algo_list, contextual_relations_list, keep_only_tokens_list)
for cutoff, context_mode, tfidf in paraml1_list:
    logging.info("Extracting context tokens for {}, {}, {}".format(cutoff, context_mode, tfidf))
    context_dict = context_per_pos(snw=semantic_network, focal_substitutes=focal_token,
                                   times=times,
                                   weight_cutoff=cutoff,
                                   keep_top_k=None, context_mode=context_mode, pos_list=pos_list, tfidf=tfidf)

    for rs,  level, depth,  max_degree, sym, keep_top_k, ma, algo, contextual_relations, keep_only_tokens in param_list:
        filename = "".join(
            [config['Paths']['csv_outputs'], "/OvrlCluster", str(focal_token), "_max_degree", str(max_degree), "_algo",
             str(algo.__name__), "_depth", str(depth), "_conRel", str(contextual_relations),
             "_depth", str(depth), "_keeptopk", str(keep_top_k), "_cm", str(context_mode), "_keeponlyt_", str(depth == 0),
             str(cutoff), "_tfidf",str(tfidf is not None), "_lev", str(level),
             str(sym), "_rs", str(rs)])
        logging.info("Overall Profiles : {}".format(filename))
        context_cluster_per_pos(semantic_network, focal_substitutes=focal_token, times=times, keep_top_k=keep_top_k,
                                context_dict =context_dict,
                                max_degree=max_degree, tfidf=tfidf, weight_cutoff=cutoff, level=level,
                                depth=depth, context_mode=context_mode, algorithm=algo,
                                contextual_relations=contextual_relations, filename=filename)


cent = get_top_100(semantic_network=semantic_network, focal_tokens=focal_token, times=times, symmetric=sym,
                   compositional=False,
                   reverse=False)
top100 = list(cent.iloc[0:100, 0].index)

filename = "".join(
    [config['Paths']['csv_outputs'], "/ListContextTokens", str(focal_token), "__", str('top100'), "_keeptopk",
     str(keep_top_k), "_cm", str(context_mode), "_rs", str(rs)])
df = context_per_pos(snw=semantic_network, focal_substitutes=focal_token, focal_occurrences=top100, times=times,
                     keep_top_k=keep_top_k, weight_cutoff=cutoff, context_mode=context_mode, filename=filename)

filename = "".join(
    [config['Paths']['csv_outputs'], "/ListContextTokens", str(focal_token), "__", str('all'), "_keeptopk",
     str(keep_top_k), "_cm", str(context_mode), "_rs", str(rs)])
df = context_per_pos(snw=semantic_network, focal_substitutes=focal_token, focal_occurrences=None, times=times,
                     keep_top_k=keep_top_k, weight_cutoff=cutoff, context_mode=context_mode, filename=filename)
