import logging
import pickle
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain, infomap_cluster, get_cluster_dict, \
    cluster_distances_from_clusterlist
from text2network.measures.context_profiles import context_cluster_all_pos, context_per_pos
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None


def get_filename(csv_path, main_folder, focal_token, cutoff, tfidf, context_mode, contextual_relations,
                 postcut, keep_top_k, depth, max_degree=None, algo=None, level=None, rs=None, tf=None, sub_mode=None):
    output_path = check_create_folder(csv_path)
    output_path = check_create_folder(csv_path + "/" + main_folder + "/")
    output_path = check_create_folder(
        "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
                 str(tfidf is not None), "_cm", str(context_mode), "/"]))
    output_path = check_create_folder("".join(
        [output_path, "/", "conRel", str(contextual_relations), "_postcut", str(int(postcut * 100)), "/"]))
    output_path = check_create_folder("".join(
        [output_path, "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "/"]))
    if max_degree is not None and algo is not None:
        output_path = check_create_folder("".join(
            [output_path, "/", "md", str(max_degree), "_algo", str(algo.__name__), "/"]))
    if sub_mode is not None:
        output_path = check_create_folder("".join(
            [output_path, "/", "submode", str(sub_mode), "/"]))
    if level is not None:
        output_path = check_create_folder("".join(
            [output_path, "/", "lev", str(level), "/"]))
    if tf is not None:
        output_path = check_create_folder("".join([output_path, "/" + "tf_", str(tf) + "/"]))
    filename = "".join(
        [output_path, "/",
         "rs", str(rs)])
    return filename, output_path


# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "6_profile_relationships.py")

semantic_network = neo4j_network(config)

# %% Across different options


times = list(range(1980, 2021))
# times=[1988]
focal_token = "leader"
sym_list = [False]
rs_list = [100]

depth_list = [0]
context_mode_list = ["occurring"]
sub_mode_list = ["occurring"]#,"substitution"]#"bidirectional"
context_mode_list = ["bidirectional"]
sub_mode_list = ["bidirectional"]#,"substitution"]#"bidirectional"
#context_mode_list = ["substitution"]
#sub_mode_list = ["substitution"]#,"substitution"]#"bidirectional"
rev_list = [False]
algo_list = [consensus_louvain]
ma_list = [(2, 2)]
pos_list = ["ADJ","VERB","NOUN","ADV","ADP"]
#pos_list = ["VERB"]
#pos_list = ["NOUN"]
# TODO CHECK WITH X
tfidf_list = [["weight"]]
level_list = [1,2,3,4,5,10,15]
keep_only_tokens_list = [True]
contextual_relations_list = [True]

keep_top_k_list = [100,200,1000]
max_degree_list = [100,200,1000]
cutoff_list = [0.1]
post_cutoff_list = [0.01]

#main_folder="profile_relationships_occ"
main_folder="profile_relationships2_"+"_".join(sub_mode_list)+"_"+"_".join(pos_list)


paraml1_list = product(cutoff_list, context_mode_list, tfidf_list)
param_list = product(rs_list, depth_list, max_degree_list, sym_list,
                     keep_top_k_list, ma_list, algo_list, contextual_relations_list, keep_only_tokens_list,
                     post_cutoff_list, sub_mode_list)

# %% Sub or Occ
#focal_substitutes = None
#focal_occurrences = focal_token
focal_substitutes = focal_token
focal_occurrences = None

if not (isinstance(focal_substitutes, list)) and not (focal_substitutes is None):
    focal_substitutes = [focal_substitutes]
if not (isinstance(focal_occurrences, list)) and not (focal_occurrences is None):
    focal_occurrences = [focal_occurrences]

# %% Extract context clusters
for cutoff, context_mode, tfidf in paraml1_list:
    logging.info("Extracting context tokens for {}, {}, {}".format(cutoff, context_mode, tfidf))

    output_path = check_create_folder(config['Paths']['csv_outputs'])
    output_path = check_create_folder(config['Paths']['csv_outputs'] + "/"+main_folder)
    cdict_filename = check_create_folder(
        "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
                 str(tfidf is not None), "_cm", str(context_mode), "/context_dict.p"]))

    try:
        logging.info("Looking for context token pickle")
        context_dict = pickle.load(open(cdict_filename, "rb"))
        logging.info("Contextual tokens loaded from {}".format(cdict_filename))
    except:
        logging.info("Context tokens pickle not found. Creating.")
        context_dict = context_per_pos(snw=semantic_network, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences,
                                       times=times,
                                       weight_cutoff=cutoff,
                                       keep_top_k=None, context_mode=context_mode, pos_list=pos_list, tfidf=tfidf)
        pickle.dump(context_dict, open(cdict_filename, "wb"))

    for rs, depth, max_degree, sym, keep_top_k, ma, algo, contextual_relations, keep_only_tokens, postcut, sub_mode in param_list:
        if postcut is None:
            postcut = 0

        if depth == 0:
            keep_only_tokens = True
            logging.info("Keeping only contextual tokens in conditioning network!")
        else:
            keep_only_tokens = False

        if keep_only_tokens and max_degree>keep_top_k:
            logging.info("Max_degree set to {}, while keep_only_tokens is set to True, and {} context_tokens are to be kept. Setting max degree to {} ".format(max_degree, keep_top_k,keep_top_k))
            max_degree = keep_top_k



        filename, output_path = get_filename(config['Paths']['csv_outputs'], main_folder,
                                             focal_token=focal_token, cutoff=cutoff, tfidf=tfidf,
                                             context_mode=context_mode, contextual_relations=contextual_relations,
                                             postcut=postcut, keep_top_k=keep_top_k, depth=depth,
                                             max_degree=None, algo=None, level=None, rs=None, tf=None, sub_mode=None)
        filename = "".join(
            [output_path, "/md", str(max_degree), "_algo", str(algo.__name__),
             "_rs", str(rs)])
        try:
            ff = check_create_folder(filename + "_clusters_raw.p")
            logging.info("Looking for Cluster dictionary pickle from {}".format(ff))
            clusters_raw = pickle.load(open(ff, "rb"))
            logging.info("Cluster dictionary loaded.")
            dfx = None
        except:
            logging.info("Cluster dictionary pickle missing, creating!")
            dfx, clusters_raw = context_cluster_all_pos(semantic_network, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
                                                        keep_top_k=keep_top_k,
                                                        max_degree=max_degree, sym=sym, weight_cutoff=postcut,
                                                        level=int(np.max(np.array(level_list))),
                                                        pos_list=pos_list, context_dict=context_dict, batch_size=10,
                                                        depth=depth, context_mode=context_mode, algorithm=algo,
                                                        include_all_levels=True,
                                                        contextual_relations=contextual_relations, tfidf=tfidf,
                                                        filename=filename + "_clustering_output")
            ff = check_create_folder(filename + "_clusters_raw.p")
            pickle.dump(clusters_raw, open(ff, "wb"))

        for level in level_list:
            logging.info("Getting level: {}".format(level))
            for tf in tfidf:
                logging.info("Getting tf-idf: {}".format(tf))
                filename, output_path = get_filename(config['Paths']['csv_outputs'], main_folder,
                                                     focal_token=focal_token, cutoff=cutoff, tfidf=tfidf,
                                                     context_mode=context_mode, contextual_relations=contextual_relations,
                                                     postcut=postcut, keep_top_k=keep_top_k, depth=depth,
                                                     max_degree=max_degree, algo=algo, level=level, rs=rs, tf=tf, sub_mode=sub_mode)

                clusters = clusters_raw[tf].copy()
                checkname = filename + "REGDF_allY" + ".xlsx"
                if os.path.exists(checkname) and os.stat(checkname).st_size > 0:
                    logging.info("File {} exists - processing completed!".format(checkname))
                else:
                    # %% Retrieve Cluster-Cluster relationships
                    logging.info("Extracted Clusters, now getting relationships")
                    clusterdict, all_nodes = get_cluster_dict(clusters, level=level, name_field="tk_top")
                    rlgraph = cluster_distances_from_clusterlist(clusters, level=level, name_field="tk_top")

                    # Calculate SimRank
                    simRank_dict = nx.simrank_similarity(rlgraph)

                    simRank_edges = [(x, y, {"weight": simRank_dict[x][y]}) for x in simRank_dict for y in simRank_dict[x] if
                                     simRank_dict[x][y] != 0.0]
                    simGraph = nx.DiGraph()
                    simGraph.add_nodes_from([x for x in simRank_dict])
                    simGraph.add_edges_from(simRank_edges)
                    simRank_edges = nx.convert_matrix.to_pandas_edgelist(simGraph)
                    simRank_edges.to_excel(filename + "_negSimRank" + ".xlsx", merge_cells=False)
                    nx.write_gexf(simGraph, path=filename + "_negSimRank" + ".gexf")

                    cl_df = nx.convert_matrix.to_pandas_adjacency(rlgraph)
                    cluster_columns = cl_df.columns
                    cl_df["type"] = "Cluster"
                    cl_df["token_id"] = list(cl_df.index)
                    cl_df["ridx"] = 0
                    cl_df["pos"] = 0
                    cl_df["year"] = 0
                    if focal_occurrences is not None:
                        cl_df.loc[:, "occ"] = "-".join(focal_occurrences)
                    else:
                        cl_df["occ"] = focal_occurrences = None
                    if focal_substitutes is not None:
                        cl_df.loc[:, "sub"] = "-".join(focal_substitutes)
                    else:
                        cl_df["sub"] = focal_substitutes = None
                    cl_df.to_excel(filename + "_CLdf" + ".xlsx", merge_cells=False)
                    # Dump cluster dict
                    pickle.dump(clusterdict, open(filename + "_CLdict.p", "wb"))
                    # Transform cluster dict to tokens
                    clusterdict_tk = clusterdict.copy()
                    for key in clusterdict_tk:
                        token_list = clusterdict_tk[key]
                        token_list = semantic_network.ensure_tokens(token_list)
                        clusterdict_tk[key] = token_list
                    pickle.dump(clusterdict_tk, open(filename + "_CLdict_tk.p", "wb"))

                    # %% Retrieve Cluster-Cluster relationships Year-On-Year
                    # logging.info("Extracting cluster relationships across years")
                    # df_list = []
                    # for year in tqdm(times, desc="YOY distances", total=len(times)):
                    #    semantic_network.decondition()
                    #    curlevel=logging.root.getEffectiveLevel()
                    #    logging.disable(logging.ERROR)
                    #    semantic_network.condition_given_dyad(dyad_substitute=focal_substitutes, dyad_occurring=focal_occurrences, times=[year],
                    #                                          focal_tokens=all_nodes, weight_cutoff=postcut, depth=0,
                    #                                          keep_only_tokens=True, batchsize=5,
                    #                                          contextual_relations=contextual_relations,
                    #                                          max_degree=max_degree)
                    #    rlgraph = cluster_distances(semantic_network.graph, clusterdict)
                    #    logging.disable(curlevel)
                    #    cl_df = nx.convert_matrix.to_pandas_edgelist(rlgraph)
                    #    cl_df["year"] = year
                    #    cl_df["type"] = "Cluster"
                    #    cl_df["token_id"] = list(cl_df.index)
                    #    if focal_occurrences is not None:
                    #        cl_df.loc[:, "occ"] = "-".join(focal_occurrences)
                    #    else:
                    #        cl_df["occ"] = focal_occurrences = None
                    #    if focal_substitutes is not None:
                    #        cl_df.loc[:, "sub"] = "-".join(focal_substitutes)
                    #    else:
                    #        cl_df["sub"] = focal_substitutes = None
                    #    cl_df["ridx"] = 0
                    #    cl_df["pos"] = 0
                    #    df_list.append(cl_df)
                    #
                    # df = pd.concat(df_list)
                    # df.to_excel(filename + "_CLdf_YOY" + ".xlsx", merge_cells=False)

                    # %% Retrieve regression data

                    allyear_list = []
                    logging.info("Substitution mode: Focal token should be {}".format(sub_mode))
                    for year in tqdm(times, desc="Iterating years", leave=False, colour='green', position=0):
                        # Step 1: get run_index, pos
                        if sub_mode == "bidirectional":
                            #query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token_id in $id_occ or a.token_id in $id_occ  and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
                            #    postcut) + " return r.pos as position, r.run_index as ridx, collect(a.token_id) as occ, collect(b.token_id) as subst, r.weight as rweight"

                            query = "MATCH p=(b:word)-[:onto]->(r:edge)-[:onto]->(a:word) WHERE b.token_id in $id_occ and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
                                postcut) + " return r.pos as position, r.run_index as ridx, collect(b.token_id) as occ, collect(a.token_id) as subst, r.weight as rweight"
                            params = {}
                            params["id_occ"] = semantic_network.ensure_ids(focal_substitutes)
                            params["times"] = [year]
                            res = semantic_network.db.receive_query(query, params)
                            occurrences = pd.DataFrame(res)

                            query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token_id in $id_occ and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
                                postcut) + " return r.pos as position, r.run_index as ridx, collect(a.token_id) as occ, collect(b.token_id) as subst, r.weight as rweight"
                            params = {}
                            params["id_occ"] = semantic_network.ensure_ids(focal_substitutes)
                            params["times"] = [year]
                            res = semantic_network.db.receive_query(query, params)
                            substitutions = pd.DataFrame(res)

                            occurrences = pd.concat([occurrences,substitutions])

                        elif sub_mode == "occurring":
                            query = "MATCH p=(b:word)-[:onto]->(r:edge)-[:onto]->(a:word) WHERE b.token_id in $id_occ and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
                                postcut) + " return r.pos as position, r.run_index as ridx, collect(b.token_id) as occ, collect(a.token_id) as subst, r.weight as rweight"
                            params = {}
                            params["id_occ"] = semantic_network.ensure_ids(focal_substitutes)
                            params["times"] = [year]
                            res = semantic_network.db.receive_query(query, params)
                            occurrences = pd.DataFrame(res)
                        else:
                            query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token_id in $id_occ and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
                                postcut) + " return r.pos as position, r.run_index as ridx, collect(a.token_id) as occ, collect(b.token_id) as subst, r.weight as rweight"
                            params = {}
                            params["id_occ"] = semantic_network.ensure_ids(focal_substitutes)
                            params["times"] = [year]
                            res = semantic_network.db.receive_query(query, params)
                            occurrences = pd.DataFrame(res)
                        #logging.info("{} Occurrences found for year {}".format(len(occurrences), year))
                        empty_dict = {x: 0 for x in cluster_columns}

                        row_list = []
                        for i, row in tqdm(occurrences.iterrows(), leave=False,
                                           desc="Iterating over occurrences in year {}".format(year), position=1, colour='red',
                                           total=len(occurrences)):
                            query = "Match (r:edge {pos:" + str(row.position) + ", run_index:" + str(
                                row.ridx) + "})-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos WITH DISTINCT r,count(DISTINCT([q.pos,q.run_index])) as seq_length Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) WHERE q.pos<>r.pos WITH DISTINCT q,r,e,seq_length RETURN q.pos as qpos, r.pos as rpos, r.run_index as ridx,sum(distinct(q.weight)) as cweight, e.token as context, head(collect(r.sentiment)) AS sentiment, head(collect(r.subjectivity)) AS subjectivity, seq_length, r.time as time order by context DESC"
                            res = pd.DataFrame(semantic_network.db.receive_query(query))
                            if len(res) > 0:
                                new_row = pd.Series(empty_dict)
                                new_row["Tyear"] = year
                                new_row["type"] = "Sub"
                                # new_row["token_id"] = list(cl_df.index)
                                new_row["occ"] = "-".join(semantic_network.ensure_tokens(row.occ))
                                new_row["sub"] = "-".join(semantic_network.ensure_tokens(row.subst))
                                new_row["ridx"] = row.ridx
                                new_row["pos"] = row.position
                                new_row["rweight"] = row.rweight
                                for cl in clusterdict_tk:
                                    tokens = semantic_network.ensure_tokens(clusterdict_tk[cl])
                                    subdf = res.loc[res.context.isin(tokens)].copy()
                                    if len(subdf) > 0:
                                        subdf.loc[:, "cweight_n"] = subdf.loc[:, "cweight"] * 40 / subdf.loc[:, "seq_length"] * row.rweight
                                        w = subdf.cweight.mean()
                                        wmin = subdf.cweight.min()
                                        wmax = subdf.cweight.max()
                                        w_n = subdf.cweight_n.mean()
                                        wmin_n = subdf.cweight_n.min()
                                        wmax_n = subdf.cweight_n.max()
                                        new_row[cl] = w_n * 100
                                        new_row["Tsent"] = subdf.sentiment.mean()
                                        new_row["Tsub"] = subdf.subjectivity.mean()
                                row_list.append(new_row)
                        yeardf = pd.DataFrame(row_list)
                        yeardf.to_excel(filename + "REGDF" + str(year) + ".xlsx", merge_cells=False)
                        allyear_list.append(yeardf)
                    allyear_df = pd.concat(allyear_list)
                    allyear_df.to_excel(filename + "REGDF_allY" + ".xlsx", merge_cells=False)
