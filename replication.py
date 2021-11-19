import logging

from scipy.spatial.distance import squareform, pdist
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from text2network.classes.neo4jnw import neo4j_network
from text2network.measures.centrality import yearly_centralities
from text2network.measures.extract_networks import extract_yearly_networks, extract_yearly_ego_networks, \
    extract_temporal_cosine_similarity
from text2network.measures.proximity import yearly_proximities
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

def get_prefix( symmetric=False, compositional=False, reverse=False,):

    prefix = ""
    if symmetric:
        prefix=prefix+"sym_"
    else:
        prefix=prefix+"fw_"
    if compositional:
        prefix=prefix+"comp_"
    if reverse:
        prefix=prefix+"rev_"

    return prefix

def fancy_dendrogram(filename, *args, title=None, **kwargs):
    """modified from joernhees.de"""
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        if title is None:
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
        else:
            plt.title(title)
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
        plt.savefig(filename, dpi=150)
        plt.show()
        plt.close()
    return ddata

def get_top_100(semantic_network,years, symmetric=False, compositional=False, reverse=False,):
    if symmetric or reverse:
        depth = 1
    else:
        depth = 0

    logging.info("Extracting Top 100 Tokens")

    semantic_network.condition(tokens=focal_token, times=years, depth=depth, prune_min_frequency=prune_min_frequency, max_degree=150)
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
    cent=cent[0:100]

    return cent


def proximities_and_centralities(semantic_network,csv_folder,years,focal_token, symmetric=False, compositional=False, reverse=False):

    prefix= get_prefix( symmetric=symmetric,compositional=compositional,reverse=reverse)

    # Create Top 100 List
    #######################
    logging.info("{}: Creating Top 100 List ".format(prefix))
    proximitiy_folder = check_create_folder(csv_folder + "/proximities")
    filename = "/" + str(prefix)+"top100_" + str(focal_token) + ".xlsx"
    ffolder = check_create_folder(proximitiy_folder + filename)

    cent=get_top_100(semantic_network=semantic_network,years=years,symmetric=symmetric,compositional=compositional,reverse=reverse)
    cent.to_excel(ffolder, merge_cells=False)
    alter_subset = list(cent.index)
    logging.info("Alter subset: {}".format(alter_subset))

    # Yearly proximities (normed by # of yearly sequences)
    #######################
    #logging.info("{}: Creating Yearly proximities (normed)".format(prefix))
    #cent = yearly_proximities(semantic_network, alter_subset=alter_subset, year_list=years, focal_tokens=focal_token,
    #                          max_degree=100, normalization="sequences", compositional=compositional, reverse=reverse, symmetric=symmetric,
    #                          symmetric_method="sum", prune_min_frequency=prune_min_frequency)
    #cent = semantic_network.pd_format(cent)[0]
    #filename = "/" + str(prefix)+"normed_proximities_" + str(focal_token) + ".xlsx"
    #ffolder = check_create_folder(proximitiy_folder + filename)
    #dyn_table = cent.reset_index(drop=False)
    #dyn_table.columns = ["year", "token", "str_eq"]
    #dyn_table.to_excel(ffolder, merge_cells=False)

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



def get_networks(semantic_network, csv_folder, years,focal_token, symmetric=False, compositional=False, reverse=False,
                  extract_yearly=True):


    prefix= get_prefix( symmetric=symmetric,compositional=compositional,reverse=reverse)



    # Extract yearly ego networks
    #######################
    logging.info("{}: Extracting Yearly Networks".format(prefix))
    network_folder = check_create_folder(csv_folder + "/yearly_networks")

    ffolder = check_create_folder(network_folder + "/" + str(prefix)+"ego")
    #extract_yearly_ego_networks(semantic_network, ego_token=focal_token, symmetric=symmetric, compositional=compositional, reverse_ties=reverse, folder=ffolder, times=years, symmetric_method="sum",
    #                        prune_min_frequency=prune_min_frequency)

    # Extract yearly networks
    #######################
    logging.info("{}: Extracting Yearly Networks".format(prefix))
    network_folder = check_create_folder(csv_folder + "/yearly_networks")

    ffolder = check_create_folder(network_folder + "/" + str(prefix))
    if extract_yearly:
        extract_yearly_networks(semantic_network, symmetric=symmetric, compositional=compositional, reverse_ties=reverse, folder=ffolder, times=years, symmetric_method="sum",
                                prune_min_frequency=prune_min_frequency)


def get_year_clusters(semantic_network, csv_folder,focal_token, years, symmetric=False, compositional=False, reverse=False):

    prefix= get_prefix( symmetric=symmetric,compositional=compositional,reverse=reverse)

    logging.info("{}: Extracting Yearly Cosine Similarities".format(prefix))

    cent=get_top_100(semantic_network=semantic_network,years=years,symmetric=symmetric,compositional=compositional,reverse=reverse)
    top100 = list(cent.manager[0:100].index)+[focal_token]
    filename = check_create_folder(csv_folder + "/yoy_clusters/" + str(prefix)+ "cosine_matrix.xlsx")

    cosine_similarities = extract_temporal_cosine_similarity(snw=semantic_network, times=years,tokens=top100,symmetric=symmetric, compositional=compositional, reverse=reverse,
                                       symmetric_method="sum", prune_min_frequency=prune_min_frequency, filename=filename)

    matrix=cosine_similarities.to_numpy()
    eucl_distances = squareform(pdist(matrix, metric='euclidean'))
    cosine_dist=1-matrix
    running_eucl_dist = squareform(pdist(np.triu(matrix), metric='euclidean'))

    linkfun_list=["complete","single","ward"]
    matrix_list={"eucl_distance":eucl_distances, "cosine_distance": cosine_dist, "running_eucl_distance": running_eucl_dist}
    df_list=[]
    for mname in matrix_list:

        dist_matrix= matrix_list[mname]
        for linkfun in linkfun_list:
            filename = check_create_folder(csv_folder + "/yoy_clusters/" + str(prefix)+ "dendogram_"+str(linkfun)+"_"+str(mname)+".pdf")
            condensed_dist_matrix = squareform(dist_matrix)
            Z = linkage(condensed_dist_matrix, linkfun)
            asdf=fancy_dendrogram(filename=filename, Z=Z,    leaf_rotation=90.,
            leaf_font_size=8., title=mname+"_"+linkfun,
            show_contracted=False,
            annotate_above=10,labels=np.array(cosine_similarities.columns))
            cluster_array=fcluster(Z, 3, criterion='maxclust')
            df=pd.DataFrame(cluster_array,index=cosine_similarities.index, columns=[mname+"_"+linkfun])
            df_list.append(df)
            for clid in np.unique(cluster_array):
                years=np.where(cluster_array==clid)[0]
                years=np.array(cosine_similarities.columns)[years]
                logging.info(str(linkfun)+"_"+str(mname)+" cluster {}, Years: {}".format(clid, years))
            del Z
    filename = check_create_folder(
        csv_folder + "/yoy_clusters/" + str(prefix) + "years_k3.xlsx")
    df=pd.concat(df_list, axis=1)
    df.to_excel(filename)


def make_regression_data(semantic_network, csv_folder, years,focal_token, symmetric=False, compositional=False, reverse=False):
    prefix= get_prefix( symmetric=symmetric,compositional=compositional,reverse=reverse)

    logging.info("{}: Extracting Regression data".format(prefix))
    top100= list(get_top_100(semantic_network=semantic_network,years=years,symmetric=symmetric,compositional=compositional,reverse=reverse).index)
    top100=top100+[focal_token]

    logging.info("{}: Creating Yearly proximities".format(prefix))
    cent = yearly_proximities(semantic_network, alter_subset=top100, year_list=years, focal_tokens=focal_token,
                              max_degree=100, normalization=None, compositional=compositional, reverse=reverse,
                              symmetric=symmetric,
                              symmetric_method="sum", prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    prox_table = cent.reset_index(drop=False)
    prox_table.columns = ["year", "token", "sym_agg_measure"]

    logging.info("{}: Creating Yearly Centralities, Clustering, Frequency  for alter subset".format(prefix))
    centrality_folder = check_create_folder(csv_folder + "/centralities")
    cent = yearly_centralities(semantic_network, year_list=years, focal_tokens=top100, symmetric_method="sum",
                               normalization=None, compositional=compositional, reverse=reverse, symmetric=symmetric,
                               types=["PageRank", "normedPageRank", "local_clustering",
                                      "frequency"], prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    cent_table = cent.reset_index(drop=False)
    cent_table.columns = ["year", "token", "sym_agg_prank", "sym_agg_nprank","unweight_trans", "freq" ]
    cent_table=cent_table[["year", "token", "sym_agg_prank", "sym_agg_nprank","unweight_trans", "freq" ]]

    mtable = pd.merge(left=prox_table, right=cent_table, how="outer", on=["year","token"])

    # Triadic Data
    logging.info("{}: Creating Yearly proximities".format(prefix))
    cent = yearly_proximities(semantic_network, alter_subset=top100, year_list=years, focal_tokens=top100,
                              max_degree=100, normalization=None, compositional=compositional, reverse=reverse,
                              symmetric=symmetric,
                              symmetric_method="sum", prune_min_frequency=prune_min_frequency)
    cent = semantic_network.pd_format(cent)[0]
    prox_table = cent.reset_index(drop=False)
    cols=prox_table.columns
    cols= [x+"_sym_agg_streq" for x in cols]
    cols[0] = "year"
    cols[1] = "token"
    prox_table.columns=cols

    mttable = pd.merge(left=mtable, right=prox_table, how="outer", on=["year","token"])
    filename = check_create_folder(
        csv_folder + "/reg_data/" + str(prefix) + "reg_table.xlsx")
    mttable.to_excel(filename)


# Set a configuration path
configuration_path = 'config/analyses/SenCMR40.ini'
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

get_networks(semantic_network=semantic_network, csv_folder=csv_folder, years=years, focal_token=focal_token, symmetric=False, compositional=False, reverse=False)
get_networks(semantic_network=semantic_network, csv_folder=csv_folder, years=years, focal_token=focal_token, symmetric=True, compositional=False, reverse=False)



make_regression_data(semantic_network=semantic_network, csv_folder=csv_folder, focal_token=focal_token, years=years, symmetric=True, compositional=False, reverse=False)
make_regression_data(semantic_network=semantic_network, csv_folder=csv_folder, focal_token=focal_token, years=years, symmetric=False, compositional=False, reverse=False)


get_year_clusters(semantic_network=semantic_network, csv_folder=csv_folder, years=years, focal_token=focal_token, symmetric=False, compositional=False, reverse=False)
get_year_clusters(semantic_network=semantic_network, csv_folder=csv_folder, years=years, focal_token=focal_token, symmetric=True, compositional=False, reverse=False)


proximities_and_centralities(semantic_network=semantic_network, csv_folder=csv_folder, focal_token=focal_token, years=years, symmetric=True, compositional=False, reverse=False)
proximities_and_centralities(semantic_network=semantic_network, csv_folder=csv_folder, focal_token=focal_token, years=years, symmetric=False, compositional=False, reverse=False)




