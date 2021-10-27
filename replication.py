import pandas as pd
import logging

from scipy.spatial.distance import squareform, pdist

from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.centrality import yearly_centralities
from text2network.measures.extract_networks import extract_yearly_networks, extract_yearly_ego_networks, \
    extract_temporal_cosine_similarity
from text2network.measures.proximity import yearly_proximities
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

def get_top_100(semantic_network,years, symmetric=False, compositional=False, reverse=False,):
    if symmetric or reverse:
        depth = 1
    else:
        depth = 0

    logging.info("Extracting Top 100 Tokens")

    semantic_network.condition(tokens=focal_token, times=years, depth=depth, prune_min_frequency=prune_min_frequency)
    # semantic_network.norm_by_total_nr_occurrences(times=years)
    if compositional:
        semantic_network.to_compositional()
    if reverse:
        semantic_network.to_reverse()
    if symmetric:
        semantic_network.to_symmetric(technique="sum")

    cent = semantic_network.proximities(focal_tokens=focal_token)
    cent = semantic_network.pd_format(cent)[0]
    cent = cent.T
    cent = cent.sort_values(by="manager", ascending=False)

    return cent


def replicate(semantic_network,csv_folder,years, symmetric=False, compositional=False, reverse=False):

    prefix = ""
    if symmetric:
        prefix=prefix+"sym_"
    if compositional:
        prefix=prefix+"comp_"
    if reverse:
        prefix=prefix+"rev_"

    # Create Top 100 List
    #######################
    logging.info("{}: Creating Top 100 List ".format(prefix))
    proximitiy_folder = check_create_folder(csv_folder + "/proximities")
    filename = "/" + str(prefix)+"top100_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(proximitiy_folder + filename)

    cent=get_top_100(semantic_network=semantic_network,years=years,symmetric=symmetric,compositional=compositional,reverse=reverse)
    cent.to_excel(ffolder, merge_cells=False)
    alter_subset = list(cent.manager[0:100].index)
    logging.info("Alter subset: {}".format(alter_subset))

    # Yearly proximities (normed by # of yearly sequences)
    #######################
    logging.info("{}: Creating Yearly proximities (normed)".format(prefix))
    cent = yearly_proximities(semantic_network, alter_subset=alter_subset, year_list=years, focal_tokens=focal_token,
                              max_degree=100, normalization="sequences", compositional=compositional, reverse=reverse, symmetric=symmetric,
                              symmetric_method="sum", prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    filename = "/" + str(prefix)+"normed_proximities_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(proximitiy_folder + filename)
    dyn_table = cent.reset_index(drop=False)
    dyn_table.columns = ["year", "token", "str_eq"]
    dyn_table.to_excel(ffolder, merge_cells=False)

    # Yearly proximities (not normed)
    #######################
    logging.info("{}: Creating Yearly proximities".format(prefix))
    cent = yearly_proximities(semantic_network, alter_subset=alter_subset, year_list=years, focal_tokens=focal_token,
                              max_degree=100, normalization=None, compositional=compositional, reverse=reverse, symmetric=symmetric,
                              symmetric_method="sum", prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    filename = "/" + str(prefix)+"proximities_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(proximitiy_folder + filename)
    dyn_table = cent.reset_index(drop=False)
    dyn_table.columns = ["year", "token", "streq"]
    dyn_table.to_excel(ffolder, merge_cells=False)

    # Network dynamics
    #######################
    logging.info("{}: Creating Network Dynamics".format(prefix))
    gg = dyn_table.groupby("token")
    counts = gg.count()['year']
    sums = gg.sum()['streq']
    streq_narm = sums / counts
    streq_avg = sums / 41
    dyn_table["first_diff"] = gg['streq'].transform(lambda x: x.diff())
    d_streq_counts = dyn_table.groupby("token")['first_diff'].count()
    d_streq_sums = dyn_table.groupby("token")['first_diff'].sum()
    d_streq_avg = d_streq_sums / d_streq_counts
    d_streq_na = 41 - d_streq_counts
    d_streq_variance = dyn_table.groupby("token")['first_diff'].var()
    d_streq_pos = (d_streq_avg > 0).astype(int)

    dp1 = dyn_table.loc[dyn_table.year == 1980, ["token", "streq"]]
    dp1 = dp1.set_index("token")
    dp2 = dyn_table.loc[dyn_table.year == 1981, ["token", "streq"]]
    dp2 = dp2.set_index("token")
    streq_avg_1980_1981 = dp2.add(dp1) / 2

    dp1 = dyn_table.loc[dyn_table.year == 2019, ["token", "streq"]]
    dp1 = dp1.set_index("token")
    dp2 = dyn_table.loc[dyn_table.year == 2020, ["token", "streq"]]
    dp2 = dp2.set_index("token")
    streq_avg_2019_2020 = dp2.add(dp1) / 2

    df = pd.concat(
        [streq_narm, streq_avg, streq_avg_1980_1981, streq_avg_2019_2020, d_streq_avg, d_streq_na, d_streq_variance,
         d_streq_pos], axis=1)
    df.columns = ["streq_avg_narm", "streq_avg", "streq_avg_1980_1981", "streq_avg_2019_2020", "d_streq_avg",
                  "d_streq_na",
                  "d_streq_variance", "d_streq_pos"]
    filename = "/" + str(prefix)+"network_dynamics_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(proximitiy_folder + filename)
    df = df.reset_index(drop=False)
    df.to_excel(ffolder, merge_cells=False)

    # Extract yearly centralities and clustering
    #######################
    logging.info("{}: Creating Yearly Centralities, Clustering, Frequency".format(prefix))
    centrality_folder = check_create_folder(csv_folder + "/centralities")
    cent = yearly_centralities(semantic_network, year_list=years, focal_tokens=focal_token, symmetric_method="sum",
                               normalization=None, compositional=compositional, reverse=reverse, symmetric=symmetric,
                               types=["PageRank", "normedPageRank", "local_clustering", "weighted_local_clustering",
                                      "frequency"], prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    filename = "/" + str(prefix)+"_centralities_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(centrality_folder + filename)
    cent.to_excel(ffolder, merge_cells=False)


    # Extract yearly centralities and clustering for alter subset
    #######################
    logging.info("{}: Creating Yearly Centralities, Clustering, Frequency  for alter subset".format(prefix))
    centrality_folder = check_create_folder(csv_folder + "/centralities")
    cent = yearly_centralities(semantic_network, year_list=years, focal_tokens=alter_subset, symmetric_method="sum",
                               normalization=None, compositional=compositional, reverse=reverse, symmetric=symmetric,
                               types=["PageRank", "normedPageRank", "local_clustering", "weighted_local_clustering",
                                      "frequency"], prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    filename = "/" + str(prefix)+"_alter_centralities_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(centrality_folder + filename)
    cent.to_excel(ffolder, merge_cells=False)

def get_networks(semantic_network, csv_folder, years, symmetric=False, compositional=False, reverse=False,
                  extract_yearly=True):


    prefix = ""
    if symmetric:
        prefix=prefix+"sym_"
    if compositional:
        prefix=prefix+"comp_"
    if reverse:
        prefix=prefix+"rev_"


    # Extract yearly ego networks
    #######################
    logging.info("{}: Extracting Yearly Networks".format(prefix))
    network_folder = check_create_folder(csv_folder + "/yearly_networks")

    ffolder = check_create_folder(network_folder + "/" + str(prefix)+"ego")
    extract_yearly_ego_networks(semantic_network, ego_token=focal_token, symmetric=symmetric, compositional=compositional, reverse_ties=reverse, folder=ffolder, times=years, symmetric_method="sum",
                            prune_min_frequency=prune_min_frequency)

    # Extract yearly networks
    #######################
    logging.info("{}: Extracting Yearly Networks".format(prefix))
    network_folder = check_create_folder(csv_folder + "/yearly_networks")

    ffolder = check_create_folder(network_folder + "/" + str(prefix))
    if extract_yearly:
        extract_yearly_networks(semantic_network, symmetric=symmetric, compositional=compositional, reverse_ties=reverse, folder=ffolder, times=years, symmetric_method="sum",
                                prune_min_frequency=prune_min_frequency)


def get_year_clusters(semantic_network, csv_folder, years, symmetric=False, compositional=False, reverse=False,
                  extract_yearly=True):

    prefix = ""
    if symmetric:
        prefix=prefix+"sym_"
    if compositional:
        prefix=prefix+"comp_"
    if reverse:
        prefix=prefix+"rev_"

    logging.info("{}: Extracting Yearly Cosine Similarities".format(prefix))

    cent=get_top_100(semantic_network=semantic_network,years=years,symmetric=symmetric,compositional=compositional,reverse=reverse)
    top100 = list(cent.manager[0:100].index)
    filename = check_create_folder(csv_folder + "/yoy_clusters/" + str(prefix)+ "cosine_matrix.xlsx")

    cosine_similarities = extract_temporal_cosine_similarity(snw=semantic_network, tokens=top100,symmetric=symmetric, compositional=compositional, reverse=reverse,
                                       symmetric_method="sum", prune_min_frequency=prune_min_frequency, filename=filename)

    matrix=cosine_similarities.to_numpy()
    eucl_distances = squareform(pdist(matrix, metric='euclidean'))

# Set a configuration path
configuration_path = 'config/analyses/replicationHBR40.ini'
# Settings
years = list(range(1980, 2021))
focal_token = "manager"
prune_min_frequency = 1

import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "replication.py")

# First, create an empty network
semantic_network = neo4j_network(config)

csv_folder = check_create_folder(config['Paths']['csv_outputs'])

get_networks(semantic_network=semantic_network, csv_folder=csv_folder, years=years, symmetric=False, compositional=False, reverse=False)
get_networks(semantic_network=semantic_network, csv_folder=csv_folder, years=years, symmetric=True, compositional=False, reverse=False)


replicate(semantic_network=semantic_network, csv_folder=csv_folder, years=years, symmetric=False, compositional=False, reverse=False)
replicate(semantic_network=semantic_network, csv_folder=csv_folder, years=years, symmetric=True, compositional=False, reverse=False)
