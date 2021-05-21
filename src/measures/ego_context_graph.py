from itertools import product
from typing import Union

from src.functions.file_helpers import check_create_folder
from src.functions.node_measures import proximity
from src.measures.measures import average_fixed_cluster_proximities, extract_all_clusters, return_measure_dict
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network
import pandas as pd
from src.classes.neo4jnw import neo4j_network
import networkx as nx

def create_ego_context_graph(snw:neo4j_network, focal_words:Union[list, str, int], ego_years:Union[list,int],context_years:Union[list,int], cluster_reduction:str="mean", max_context_clusters:int=10, max_ego_clusters:int=10,symmetric:bool=False, compositional:bool=False, cutoff:float=0, level:int=7, scaling_factor:float=100.0 )->Union[nx.Graph, nx.DiGraph]:

    logging.info("Finding context ties of focal tokens.")
    snw.decondition()
    snw.context_condition(times=context_years, tokens=focal_words, depth=1, weight_cutoff=cutoff)  # condition on context
    if symmetric: snw.to_symmetric()

    # Get clusters
    clusters = snw.cluster(levels=level, to_measure=[proximity], algorithm=consensus_louvain)
    dataframe_list = []
    context_level_list = []
    for cl in clusters:
        if cl['level'] == 0:
            c_proxim = snw.pd_format(cl['measures'])[0]
            c_proxim_np = snw.pd_format(cl['measures'], ids_to_tokens=False)[0]
            nodes = list(c_proxim_np.index)
        if cl['level'] == level:
            context_level_list.append(list(cl['graph'].nodes))
            cl_nodes = list(cl['graph'].nodes)
            focal_proxim = c_proxim.loc[focal_words[0], :].reindex(snw.ensure_tokens(cl_nodes), fill_value=0)
            cluster_measures = return_measure_dict(focal_proxim)
            top_node = focal_proxim.idxmax() if len(focal_proxim) > 0 else ""
            name = "-".join(list(focal_proxim.nlargest(5).index))
            df_dict = {'Token': name, 'Level': cl['level'], 'Clustername': cl['name'], 'Prom_Node': top_node,
                       'Parent': cl['parent'], 'Nr_ProxNodes': len(focal_proxim), 'NrNodes': len(cl_nodes)}
            df_dict.update(cluster_measures)
            dataframe_list.append(df_dict.copy())

    context_clusters = pd.DataFrame(dataframe_list)
    context_clusters = context_clusters.sort_values(by="w_Avg", ascending=False)

    context_clusters = context_clusters.iloc[:max_context_clusters,:]
    context_clusters_ids = list(context_clusters.index)

    context_graph=nx.DiGraph()
    for i, j in product(context_clusters_ids, context_clusters_ids):

        if cluster_reduction=="max":
            weight = c_proxim_np.loc[context_level_list[i], context_level_list[j]].max().max() * scaling_factor
            if not symmetric:
                r_weight = c_proxim_np.loc[context_level_list[j], context_level_list[i]].max().max() * scaling_factor
        else:
            weight= c_proxim_np.loc[context_level_list[i], context_level_list[j]].mean().mean() * scaling_factor
            if not symmetric:
                r_weight = c_proxim_np.loc[context_level_list[j], context_level_list[i]].mean().mean() * scaling_factor

        i_token="c_"+context_clusters.loc[[i]].Prom_Node.values[0]
        j_token ="c_"+context_clusters.loc[[j]].Prom_Node.values[0]
        context_graph.add_nodes_from([i_token,j_token])
        if i_token != j_token:
            edge=(i_token,j_token,{'weight':weight, 'edgetype':"context"})
            context_graph.add_edges_from([edge])
            if not symmetric:
                edge = (j_token,i_token, {'weight': r_weight, 'edgetype': "context"})
                context_graph.add_edges_from([edge])


    context_nodes = list(context_clusters.Prom_Node)

    # Get ego clusters
    logging.info("Getting ego network clusters.")
    snw.decondition()
    snw.condition(times=ego_years, tokens=focal_words, depth=1, weight_cutoff=cutoff,
                  compositional=compositional)
    if symmetric: snw.to_symmetric()

    clusters = snw.cluster(levels=level, to_measure=[proximity], algorithm=consensus_louvain)
    dataframe_list = []
    level_list_ego=[]
    for cl in clusters:
        if cl['level'] == 0:
            ego_proxim = snw.pd_format(cl['measures'])[0]
            ego_proxim_np = snw.pd_format(cl['measures'], ids_to_tokens=False)[0]
            nodes = list(ego_proxim.index)
        if cl['level'] == level:
            level_list_ego.append(list(cl['graph'].nodes))
            cl_nodes = list(cl['graph'].nodes)
            focal_proxim = snw.pd_format(snw.proximities(focal_tokens=focal_words, alter_subset=cl_nodes))[0].loc[
                           focal_words[0], :]
            print(cl_nodes)
            print(focal_proxim)
            cluster_measures = return_measure_dict(focal_proxim)
            top_node = focal_proxim.idxmax() if len(focal_proxim) > 0 else ""
            name = "-".join(list(focal_proxim.nlargest(5).index))
            if top_node is not "":
                df_dict = {'Token': name, 'Level': cl['level'], 'Clustername': cl['name'], 'Prom_Node': top_node,
                           'Parent': cl['parent'], 'Nr_ProxNodes': len(focal_proxim), 'NrNodes': len(cl_nodes)}
                df_dict.update(cluster_measures)
                dataframe_list.append(df_dict.copy())

    ego_clusters = pd.DataFrame(dataframe_list)
    ego_clusters = ego_clusters.sort_values(by="w_Avg", ascending=False)
    ego_clusters = ego_clusters.iloc[:max_ego_clusters, :]

    ego_cluster_ids = list(ego_clusters.index)
    nr_clusters = len(ego_cluster_ids)
    ego_graph = nx.DiGraph()
    for i, j in product(ego_cluster_ids, ego_cluster_ids):

        if cluster_reduction == "max":
            weight = ego_proxim_np.loc[level_list_ego[i], level_list_ego[j]].max().max() * scaling_factor
            if not symmetric:
                r_weight = ego_proxim_np.loc[context_level_list[j], context_level_list[i]].max().max() * scaling_factor
        else:
            weight = ego_proxim_np.loc[level_list_ego[i], level_list_ego[j]].mean().mean() * scaling_factor
            if not symmetric:
                r_weight = c_proxim_np.loc[context_level_list[j], context_level_list[i]].mean().mean() * scaling_factor

        i_token = ego_clusters.loc[[i]].Prom_Node.values[0]
        j_token = ego_clusters.loc[[j]].Prom_Node.values[0]
        ego_graph.add_nodes_from([i_token, j_token])
        if i_token != j_token:
            edge = (i_token, j_token, {'weight': weight, 'edgetype': "ego"})
            ego_graph.add_edges_from([edge])
            if not symmetric:
                edge = (j_token,i_token, {'weight': r_weight, 'edgetype': "ego"})
                ego_graph.add_edges_from([edge])

    ego_nodes = list(ego_clusters.Prom_Node)


    # Ego to context
    logging.info("Calculating ties between context and ego network")
    snw.decondition()
    snw.context_condition(times=ego_years, tokens=ego_nodes + context_nodes, depth=1, weight_cutoff=cutoff)

    context_ego_graph=ego_graph.copy()
    context_ego_graph.add_edges_from(context_graph.edges)

    for i, j in product(ego_cluster_ids, context_clusters_ids):

        print(i)
        print(j)
        ego_subset=level_list_ego[i]
        context_subset=context_level_list[j]
        ego_subset_name=ego_clusters.loc[[i]].Prom_Node.values[0]
        context_subset_name="c_"+context_clusters.loc[[j]].Prom_Node.values[0]
        #snw.decondition()
        #snw.context_condition(times=ego_years, tokens=ego_subset+context_subset, depth=1, weight_cutoff=cutoff)
        proxim_np = snw.pd_format(snw.proximities(focal_tokens=ego_subset+context_subset, alter_subset=ego_subset+context_subset), ids_to_tokens=False)[0]
        ego_to_context=proxim_np.reindex(ego_subset, fill_value=0).reindex(context_subset, fill_value=0, axis=1)
        context_to_ego =proxim_np.reindex(context_subset, fill_value=0).reindex(ego_subset, fill_value=0, axis=1)
        if cluster_reduction == "max":
            weight = ego_to_context.max().max() * scaling_factor
            if symmetric:
                r_weight= context_to_ego.max().max() * scaling_factor
        else:
            weight = ego_to_context.mean().mean() * scaling_factor
            if symmetric:
                r_weight= context_to_ego.mean().mean() * scaling_factor
        if ego_subset_name != context_subset_name:
            edge = (ego_subset_name, context_subset_name, {'weight': weight, 'edgetype': "egocontext"})
            context_ego_graph.add_edges_from([edge])
            if not symmetric:
                edge = (context_subset_name,ego_subset_name, {'weight': r_weight, 'edgetype': "egocontext"})
                context_ego_graph.add_edges_from([edge])


    # Set hierarchy
    hierarchy_dict = {}
    for node in context_ego_graph.nodes():
        if node in context_graph.nodes():
            hierarchy_dict[node] = {"hierarchy": 0}
        else:
            hierarchy_dict[node] = {"hierarchy": 1}
    nx.set_node_attributes(context_ego_graph, hierarchy_dict)


    return context_ego_graph