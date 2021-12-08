import logging
import pickle
from itertools import product

import networkx as nx
import pandas as pd
from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain, cluster_distances, \
    cluster_distances_from_clusterlist, get_cluster_dict, infomap_cluster
from text2network.measures.context_profiles import context_cluster_all_pos, context_per_pos
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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "6_profile_relationships.py")

semantic_network = neo4j_network(config)

times = list(range(1980, 1982))
focal_token = "leader"
sym = False
keep_top_k = 100
max_degree = 100
rs = 100
cutoff = 0.2
level = 1
depth = 0
context_mode = "bidirectional"
rev = False
algo = consensus_louvain
ma = (2, 2)
keep_only_tokens = True
contextual_relations = True
pos_list = ["NOUN", "VERB", "ADJ"]
tfidf = ["rel_weight", "pmi_weight","weight","diff", "diffw" ]


# %% Across different options


times = list(range(1980, 2021))
times=[1990]

focal_token = "leader"
sym_list = [False]
rs_list = [100]
cutoff_list = [0.2,0.1,0.01]
depth_list = [0, 1]
context_mode_list = ["bidirectional", "substitution", "occurring"]
rev_list = [False]
algo_list = [consensus_louvain, infomap_cluster]
ma_list = [(2, 2)]
pos_list = ["NOUN", "ADJ", "VERB"]
# TODO CHECK WITH X
#pos_list = ["X"]
tfidf_list = [["rel_weight", "pmi_weight","weight","diff", "diffw" ]]
keep_top_k_list = [50,100,200,1000]
max_degree_list = [50,100]
level_list = [2,3,6,15]
keep_only_tokens_list = [True,False]
contextual_relations_list = [True,False]

paraml1_list = product(cutoff_list,context_mode_list,tfidf_list)
param_list = product(rs_list,  level_list, depth_list,  max_degree_list, sym_list,
                     keep_top_k_list, ma_list, algo_list, contextual_relations_list, keep_only_tokens_list)





# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes=[focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences=[focal_occurrences]

# %% Extract context clusters


for cutoff, context_mode, tfidf in paraml1_list:
    logging.info("Extracting context tokens for {}, {}, {}".format(cutoff, context_mode, tfidf))
    context_dict = context_per_pos(snw=semantic_network, focal_substitutes=focal_substitutes,
                                   times=times,
                                   weight_cutoff=cutoff,
                                   keep_top_k=None, context_mode=context_mode, pos_list=pos_list, tfidf=tfidf)


    for rs, level, depth, max_degree, sym, keep_top_k, ma, algo, contextual_relations, keep_only_tokens in param_list:
        output_path = check_create_folder(config['Paths']['csv_outputs'])
        output_path = check_create_folder(config['Paths']['csv_outputs'] + "/profile_relationships")
        output_path = check_create_folder("".join(
            [output_path, "/", str(focal_token), "_conRel", str(contextual_relations), "_cm", str(context_mode), "/"]))
        output_path = check_create_folder("".join(
            [output_path, "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "_tfidf",
             str(tfidf is not None), "_cut", str(int(cutoff * 100)), "_lev", str(level), "/"]))

        filename = check_create_folder("".join(
            [output_path, "/OvrlCluster_", "-".join(pos_list),"_max_degree", str(max_degree), "_algo",
             str(algo.__name__),
             "_rs", str(rs)]))
        logging.info("Overall Profiles  Regression tables: {}".format(filename))

        df, clusters_raw = context_cluster_all_pos(semantic_network, focal_substitutes=focal_token, times=times, keep_top_k=keep_top_k,
                                max_degree=max_degree, sym=sym, weight_cutoff=cutoff, level=level,
                                pos_list=pos_list, context_dict =context_dict,
                                depth=depth, context_mode=context_mode, algorithm=algo,
                                contextual_relations=contextual_relations, tfidf=tfidf, filename=filename)

        for tf in tfidf:
            clusters=clusters_raw[tf].copy()
            filename =  check_create_folder("".join(
                [output_path, "/OvrlCluster_", "-".join(pos_list),"_tf", str(tf), "_max_degree", str(max_degree), "_algo", str(algo.__name__),
                 "_rs", str(rs)]))


            # %% Retrieve Cluster-Cluster relationships
            logging.info("Extracted Clusters, now getting relationships")
            clusterdict, all_nodes = get_cluster_dict(clusters, level=level, name_field="tk_top")
            rlgraph = cluster_distances_from_clusterlist(clusters, level=level, name_field="tk_top")

            # Calculate SimRank
            simRank_dict = nx.simrank_similarity(rlgraph)

            simRank_edges= [(x,y,{"weight":simRank_dict[x][y]}) for x in simRank_dict for y in simRank_dict[x] if simRank_dict[x][y] != 0.0]
            simGraph = nx.DiGraph()
            simGraph.add_nodes_from([x for x in simRank_dict])
            simGraph.add_edges_from(simRank_edges)
            simRank_edges=nx.convert_matrix.to_pandas_edgelist(simGraph)
            simRank_edges.to_excel(filename + "_negSimRank" + ".xlsx", merge_cells=False)
            nx.write_gexf(simGraph, path=filename + "_negSimRank" + ".gexf")



            cl_df = nx.convert_matrix.to_pandas_adjacency(rlgraph)
            cluster_columns=cl_df.columns
            cl_df["type"]="Cluster"
            cl_df["token_id"]=list(cl_df.index)
            cl_df["ridx"]=0
            cl_df["pos"]=0
            cl_df["year"]=0
            if focal_occurrences is not None:
                cl_df.loc[:,"occ"]="-".join(focal_occurrences)
            else:
                cl_df["occ"]=focal_occurrences = None
            if focal_substitutes is not None:
                cl_df.loc[:,"sub"]="-".join(focal_substitutes)
            else:
                cl_df["sub"]=focal_substitutes = None
            cl_df.to_excel(filename + "_CLdf" + ".xlsx", merge_cells=False)
            # Dump cluster dict
            pickle.dump(clusterdict, open(filename + "_CLdict.p", "wb"))
            # Transform cluster dict to tokens
            clusterdict_tk = clusterdict.copy()
            for key in clusterdict_tk:
                token_list=clusterdict_tk[key]
                token_list=semantic_network.ensure_tokens(token_list)
                clusterdict_tk[key]=token_list
            pickle.dump(clusterdict_tk, open(filename + "_CLdict_tk.p", "wb"))

            # %% Retrieve Cluster-Cluster relationships Year-On-Year
            logging.info("Extracting cluster relationships across years")
            df_list = []
            for year in times:
                semantic_network.decondition()
                semantic_network.condition_given_dyad(dyad_substitute=focal_substitutes, dyad_occurring=focal_occurrences, times=[year],
                                                      focal_tokens=all_nodes, weight_cutoff=cutoff, depth=0,
                                                      keep_only_tokens=True,
                                                      contextual_relations=contextual_relations,
                                                      max_degree=max_degree)
                rlgraph = cluster_distances(semantic_network.graph, clusterdict)
                cl_df = nx.convert_matrix.to_pandas_edgelist(rlgraph)
                cl_df["year"] = year
                cl_df["type"] = "Cluster"
                cl_df["token_id"] = list(cl_df.index)
                if focal_occurrences is not None:
                    cl_df.loc[:, "occ"] = "-".join(focal_occurrences)
                else:
                    cl_df["occ"] = focal_occurrences = None
                if focal_substitutes is not None:
                    cl_df.loc[:, "sub"] = "-".join(focal_substitutes)
                else:
                    cl_df["sub"] = focal_substitutes = None
                cl_df["ridx"] = 0
                cl_df["pos"] = 0
                df_list.append(cl_df)

            df = pd.concat(df_list)
            df.to_excel(filename + "_CLdf_YOY" + ".xlsx", merge_cells=False)


            # %% Retrieve regression data

            allyear_list=[]
            for year in tqdm(times, desc="Iterating years", leave=False, colour='green', position=0):
                # Step 1: get run_index, pos
                query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token_id in $id_occ and a.token_id <> b.token_id and r.time in $times return r.pos as position, r.run_index as ridx, collect(a.token_id) as occ, collect(b.token_id) as subst, r.weight as rweight"
                params={}
                params["id_occ"]=semantic_network.ensure_ids(focal_substitutes)
                params["times"]=[year]
                res=semantic_network.db.receive_query(query, params)
                occurrences=pd.DataFrame(res)
                logging.info("{} Occurrences found for year {}".format(len(occurrences), year))
                empty_dict = {x:0 for x in cluster_columns}

                row_list=[]
                for i, row in tqdm(occurrences.iterrows(),leave=False, desc="Iterating over occurrences in year {}".format(year),position=1, colour='red', total=len(occurrences)):

                    query="Match (r:edge {pos:"+ str(row.position) +", run_index:"+ str(row.ridx)+ "})-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos WITH DISTINCT r,count(DISTINCT([q.pos,q.run_index])) as seq_length Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) WHERE q.pos<>r.pos WITH DISTINCT q,r,e,seq_length RETURN q.pos as qpos, r.pos as rpos, r.run_index as ridx,sum(distinct(q.weight)) as cweight, e.token as context, head(collect(r.sentiment)) AS sentiment, head(collect(r.subjectivity)) AS subjectivity, seq_length, r.time as time order by context DESC"
                    res = pd.DataFrame(semantic_network.db.receive_query(query))

                    new_row =  pd.Series(empty_dict)
                    new_row["year"] = year
                    new_row["type"] = "Sub"
                    #new_row["token_id"] = list(cl_df.index)
                    new_row["occ"] = "-".join(semantic_network.ensure_tokens(row.occ))
                    new_row["sub"] = "-".join(semantic_network.ensure_tokens(row.subst))
                    new_row["ridx"] = row.ridx
                    new_row["pos"] = row.position
                    new_row["rweight"] = row.rweight
                    for cl in clusterdict_tk:
                        tokens=clusterdict_tk[cl]
                        subdf=res.loc[res.context.isin(tokens)].copy()
                        if len(subdf)>0:
                            subdf.loc[:,"cweight_n"]=subdf.loc[:,"cweight"] * 40/subdf.loc[:,"seq_length"] * row.rweight
                            w=subdf.cweight.mean()
                            wmin = subdf.cweight.min()
                            wmax = subdf.cweight.max()
                            w_n=subdf.cweight_n.mean()
                            wmin_n = subdf.cweight_n.min()
                            wmax_n = subdf.cweight_n.max()
                            new_row[cl] = w_n*100
                    row_list.append(new_row)
                yeardf=pd.DataFrame(row_list)
                yeardf.to_excel(filename + "REGDF" + str(year) +".xlsx", merge_cells=False)
                allyear_list.append(yeardf)
            allyear_df=pd.concat(allyear_list)
            allyear_df.to_excel(filename + "REGDF_allY_" + ".xlsx", merge_cells=False)


#df, clusters = context_cluster_all_pos(semantic_network, focal_substitutes=focal_substitutes, focal_occurrences=focal_occurrences, times=times,
#                                       keep_top_k=keep_top_k, max_degree=max_degree, sym=sym, weight_cutoff=cutoff,
#                                       level=level, include_all_levels=False,
#                                       depth=depth, context_mode=context_mode, algorithm=algo,
#                                       contextual_relations=contextual_relations, pos_list=pos_list, tfidf=tfidf,
#                                       filename=None)