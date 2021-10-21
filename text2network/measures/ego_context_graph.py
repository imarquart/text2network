from itertools import product
from typing import Union, Optional

from src.functions.file_helpers import check_create_folder
from src.functions.node_measures import proximity
from src.measures.measures import average_cluster_proximities, extract_all_clusters, return_measure_dict
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain
from src.classes.neo4jnw import neo4j_network
import pandas as pd
import networkx as nx

from src.utils.rowvec_tools import cutoff_percentage


def create_ego_context_graph(snw: neo4j_network, focal_words: Union[list, str, int], ego_years: Union[list, int],
                             context_years: Union[list, int], cluster_reduction: str = "mean",
                             moving_average: Optional[tuple] = None,
                             max_context_clusters: int = 10, max_ego_clusters: int = 10, symmetric: bool = False,
                             compositional: bool = False, ego_cutoff: float = 0, context_cutoff: float = 0,
                             level: int = 7, scaling_factor: float = 100.0, interest_list: list = None) -> Union[
    nx.Graph, nx.DiGraph]:
    if not isinstance(focal_words, list):
        focal_words = [focal_words]

    if not isinstance(ego_years, list):
        ego_years = [ego_years]

    logging.info("Finding context ties of focal tokens.")
    snw.decondition()
    snw.context_condition(times=context_years, tokens=focal_words, depth=1,
                          weight_cutoff=context_cutoff)  # condition on context
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

    context_clusters = context_clusters.iloc[:max_context_clusters, :]
    context_clusters_ids = list(context_clusters.index)

    context_graph = nx.DiGraph()
    for i, j in product(context_clusters_ids, context_clusters_ids):

        if cluster_reduction == "max":
            weight = c_proxim_np.loc[context_level_list[i], context_level_list[j]].max().max() * scaling_factor
            if not symmetric:
                r_weight = c_proxim_np.loc[context_level_list[j], context_level_list[i]].max().max() * scaling_factor
        else:
            weight = c_proxim_np.loc[context_level_list[i], context_level_list[j]].mean().mean() * scaling_factor
            if not symmetric:
                r_weight = c_proxim_np.loc[context_level_list[j], context_level_list[i]].mean().mean() * scaling_factor

        i_token = "c_" + context_clusters.loc[[i]].Prom_Node.values[0]
        j_token = "c_" + context_clusters.loc[[j]].Prom_Node.values[0]
        i_desc = context_clusters.loc[[i]].Token.values[0]
        j_desc = context_clusters.loc[[j]].Token.values[0]
        context_graph.add_nodes_from([i_token, j_token])

        name_dict = {}
        name_dict[i_token] = {"desc": i_desc}
        name_dict[j_token] = {"desc": j_desc}
        nx.set_node_attributes(context_graph, name_dict)

        if i_token != j_token:
            edge = (i_token, j_token, {'weight': weight, 'edgetype': "context"})
            if weight > 0:
                context_graph.add_edges_from([edge])
            if not symmetric:
                edge = (j_token, i_token, {'weight': r_weight, 'edgetype': "context"})
                if r_weight > 0:
                    context_graph.add_edges_from([edge])

    context_nodes = list(context_clusters.Prom_Node)

    # Get ego clusters
    logging.info("Getting ego network clusters across years {}.".format(ego_years))
    snw.decondition()
    snw.condition(times=ego_years, tokens=focal_words, depth=1, weight_cutoff=ego_cutoff,
                  compositional=compositional)
    if symmetric: snw.to_symmetric()

    clusters = snw.cluster(levels=level, to_measure=[proximity], algorithm=consensus_louvain,
                           interest_list=interest_list)
    dataframe_list = []
    level_list_ego = []
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
    ego_nodes = list(ego_clusters.Prom_Node)

    logging.info("Populating year-on-year ego networks.")

    ego_graph = nx.DiGraph()
    for year in ego_years:
        snw.decondition()

        if moving_average is not None:
            start_year = max(ego_years[0], year - moving_average[0])
            end_year = min(ego_years[-1], year + moving_average[1])
            ma_years = np.arange(start_year, end_year + 1)
        else:
            ma_years = year
        logging.info(
            "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                year,
                moving_average[
                    0],
                moving_average[
                    1], ma_years))
        snw.condition(times=ma_years, tokens=ego_nodes, depth=1, weight_cutoff=ego_cutoff,
                      compositional=compositional)
        if symmetric: snw.to_symmetric()
        ego_proxim_np = snw.pd_format(
            snw.proximities(focal_tokens=ego_nodes, alter_subset=ego_nodes),
            ids_to_tokens=False)[0]

        for i, j in product(ego_cluster_ids, ego_cluster_ids):

            if cluster_reduction == "max":
                weight = ego_proxim_np.reindex(level_list_ego[i], axis=0, fill_value=0).reindex(level_list_ego[j],
                                                                                                axis=1,
                                                                                                fill_value=0).max().max() * scaling_factor
                if not symmetric:
                    r_weight = ego_proxim_np.reindex(level_list_ego[j], axis=0, fill_value=0).reindex(level_list_ego[i],
                                                                                                      axis=1,
                                                                                                      fill_value=0).max().max() * scaling_factor
            else:
                weight = ego_proxim_np.reindex(level_list_ego[i], axis=0, fill_value=0).reindex(level_list_ego[j],
                                                                                                axis=1,
                                                                                                fill_value=0).mean().mean() * scaling_factor
                if not symmetric:
                    r_weight = ego_proxim_np.reindex(level_list_ego[j], axis=0, fill_value=0).reindex(level_list_ego[i],
                                                                                                      axis=1,
                                                                                                      fill_value=0).mean().mean() * scaling_factor

            i_token = str(year) + "_" + ego_clusters.loc[[i]].Prom_Node.values[0]
            j_token = str(year) + "_" + ego_clusters.loc[[j]].Prom_Node.values[0]

            i_desc = ego_clusters.loc[[i]].Token.values[0]
            j_desc = ego_clusters.loc[[j]].Token.values[0]

            name_dict = {}
            name_dict[i_token] = {"desc": i_desc}
            name_dict[j_token] = {"desc": j_desc}
            ego_graph.add_nodes_from([i_token, j_token])
            nx.set_node_attributes(ego_graph, name_dict)
            if i_token != j_token:
                edge = (i_token, j_token, {'weight': weight, 'edgetype': "ego"})
                if weight > 0:
                    ego_graph.add_edges_from([edge])
                if not symmetric:
                    edge = (j_token, i_token, {'weight': r_weight, 'edgetype': "ego"})
                    if r_weight > 0:
                        ego_graph.add_edges_from([edge])
        name_dict = {}
        for node in ego_nodes:
            c_node = str(year) + "_" + node
            name_dict[c_node] = {"token": node, "year": year}
        nx.set_node_attributes(ego_graph, name_dict)

    context_ego_graph = ego_graph.copy()
    context_ego_graph.add_nodes_from(context_graph.nodes.data())
    context_ego_graph.add_edges_from(context_graph.edges.data())

    logging.info("Calculating ties between context and ego network")
    for year in ego_years:
        logging.info("Conditioning context on year {}.".format(year))
        # Ego to context
        snw.decondition()
        snw.context_condition(times=year, tokens=ego_nodes + context_nodes, depth=1, weight_cutoff=context_cutoff)
        for i, j in product(ego_cluster_ids, context_clusters_ids):

            ego_subset = level_list_ego[i]
            context_subset = context_level_list[j]
            ego_subset_name = str(year) + "_" + ego_clusters.loc[[i]].Prom_Node.values[0]
            context_subset_name = "c_" + context_clusters.loc[[j]].Prom_Node.values[0]

            proxim_np = snw.pd_format(
                snw.proximities(focal_tokens=ego_subset + context_subset, alter_subset=ego_subset + context_subset),
                ids_to_tokens=False)[0]
            ego_to_context = proxim_np.reindex(ego_subset, fill_value=0).reindex(context_subset, fill_value=0, axis=1)
            context_to_ego = proxim_np.reindex(context_subset, fill_value=0).reindex(ego_subset, fill_value=0, axis=1)
            if cluster_reduction == "max":
                weight = ego_to_context.max().max() * scaling_factor
                if symmetric:
                    r_weight = context_to_ego.max().max() * scaling_factor
            else:
                weight = ego_to_context.mean().mean() * scaling_factor
                if symmetric:
                    r_weight = context_to_ego.mean().mean() * scaling_factor
            if ego_subset_name != context_subset_name:
                edge = (ego_subset_name, context_subset_name, {'weight': weight, 'edgetype': "egocontext"})
                if weight > 0:
                    context_ego_graph.add_edges_from([edge])
                if not symmetric:
                    edge = (context_subset_name, ego_subset_name, {'weight': r_weight, 'edgetype': "egocontext"})
                    if r_weight > 0:
                        context_ego_graph.add_edges_from([edge])
                    context_ego_graph.add_edges_from([edge])

    # Set hierarchy
    hierarchy_dict = {}
    for node in context_ego_graph.nodes():
        if node in context_graph.nodes():
            hierarchy_dict[node] = {"hierarchy": 0}
        else:
            hierarchy_dict[node] = {"hierarchy": 1}
    nx.set_node_attributes(context_ego_graph, hierarchy_dict)

    edge_weights = nx.get_edge_attributes(context_ego_graph, 'weight')
    context_ego_graph.remove_edges_from((e for e, w in edge_weights.items() if w <= 0))
    isolates = list(nx.isolates(context_ego_graph))
    logging.debug(
        "Found {} isolated nodes in graph, deleting.".format(len(isolates)))
    context_ego_graph.remove_nodes_from(isolates)

    return context_ego_graph


def create_ego_context_graph_simple(snw: neo4j_network, focal_word: Union[str, int], replacement_cluster_list: list,
                                    context_cluster_list: list, ego_years: Union[list, int],
                                    context_years: Union[list, int], cluster_reduction: str = "mean",
                                    moving_average: Optional[tuple] = None,
                                    symmetric: bool = False, sparsity_percentage=99,
                                    compositional: bool = False, ego_cutoff: float = 0, context_cutoff: float = 0,
                                    scaling_factor: float = 100.0) -> Union[nx.Graph, nx.DiGraph]:
    if not isinstance(focal_word, list):
        focal_word = [focal_word]

    if not isinstance(ego_years, list):
        ego_years = [ego_years]

    ego_tokens = list(set([item for sublist in replacement_cluster_list for item in sublist]))
    ego_tokens.extend(focal_word)
    context_tokens = list(set([item for sublist in context_cluster_list for item in sublist]))
    context_tokens.extend(focal_word)

    logging.info("Finding context ties of focal tokens.")
    snw.decondition()
    snw.context_condition(times=context_years, tokens=focal_word, depth=1,
                          weight_cutoff=context_cutoff)  # condition on context
    if symmetric: snw.to_symmetric()
    if sparsity_percentage is not None:
        snw.sparsify(sparsity_percentage)
    # Get all names

    # Get proximities
    context_proximities = snw.pd_format(snw.proximities(focal_tokens=context_tokens, alter_subset=context_tokens))[0]
    #context_proximities = cutoff_percentage(context_proximities, sparsity_percentage)
    # Create context cluster descriptions
    dataframe_list = []
    for cluster in context_cluster_list:
        cluster = snw.ensure_tokens(cluster)

        focal_proxim = context_proximities.loc[focal_word[0], :].reindex(cluster, fill_value=0)
        cluster_measures = return_measure_dict(focal_proxim)
        top_node = focal_proxim.idxmax() if len(focal_proxim) > 0 else ""
        name = "-".join(list(focal_proxim.nlargest(5).index))
        df_dict = {'Token': name, 'Prom_Node': top_node, 'Nr_ProxNodes': len(focal_proxim), 'NrNodes': len(cluster)}
        df_dict.update(cluster_measures)
        dataframe_list.append(df_dict.copy())

    context_clusters = pd.DataFrame(dataframe_list)
    context_clusters = context_clusters.sort_values(by="w_Avg", ascending=False)
    context_clusters_ids = list(context_clusters.index)

    # Create context graph
    context_graph = nx.DiGraph()
    for i, j in product(context_clusters_ids, context_clusters_ids):
        if i != j:
            if cluster_reduction == "max":
                weight = context_proximities.reindex(context_cluster_list[i], axis=0, fill_value=0).reindex(context_cluster_list[j], axis=1, fill_value=0).max().max()  * scaling_factor
                if not symmetric:
                    r_weight = context_proximities.reindex(context_cluster_list[j], axis=0, fill_value=0).reindex(context_cluster_list[i], axis=1, fill_value=0).max().max()  * scaling_factor
            else:
                weight = context_proximities.reindex(context_cluster_list[i], axis=0, fill_value=0).reindex(context_cluster_list[j], axis=1, fill_value=0).mean().mean() * scaling_factor
                if not symmetric:
                    r_weight = context_proximities.reindex(context_cluster_list[j], axis=0, fill_value=0).reindex(context_cluster_list[i], axis=1, fill_value=0).mean().mean()  * scaling_factor

            i_token = "c_" + context_clusters.loc[[i]].Prom_Node.values[0]
            j_token = "c_" + context_clusters.loc[[j]].Prom_Node.values[0]
            i_desc = context_clusters.loc[[i]].Token.values[0]
            j_desc = context_clusters.loc[[j]].Token.values[0]
            context_graph.add_nodes_from([i_token, j_token])

            name_dict = {}
            name_dict[i_token] = {"desc": i_desc}
            name_dict[j_token] = {"desc": j_desc}
            nx.set_node_attributes(context_graph, name_dict)

            if i_token != j_token:
                edge = (i_token, j_token, {'weight': weight, 'edgetype': "context"})
                if weight > 0:
                    context_graph.add_edges_from([edge])
                if not symmetric:
                    edge = (j_token, i_token, {'weight': r_weight, 'edgetype': "context"})
                    if r_weight > 0:
                        context_graph.add_edges_from([edge])

    context_prom_nodes = list(context_clusters.Prom_Node)

    # Get ego cluster descriptions
    logging.info("Getting ego network clusters across years {}.".format(ego_years))
    snw.decondition()
    snw.condition(times=ego_years, tokens=focal_word, depth=1, weight_cutoff=ego_cutoff,
                  compositional=compositional)
    if symmetric: snw.to_symmetric()
    if sparsity_percentage is not None:
        snw.sparsify(sparsity_percentage)
    # Get all names

    # Get proximities
    ego_proximities = snw.pd_format(snw.proximities(focal_tokens=context_tokens, alter_subset=context_tokens))[0]
    #ego_proximities = cutoff_percentage(ego_proximities, sparsity_percentage)
    # Create context cluster descriptions
    dataframe_list = []
    for cluster in replacement_cluster_list:
        cluster = snw.ensure_tokens(cluster)

        focal_proxim = ego_proximities.loc[focal_word[0], :].reindex(cluster, fill_value=0)
        cluster_measures = return_measure_dict(focal_proxim)
        top_node = focal_proxim.idxmax() if len(focal_proxim) > 0 else ""
        name = "-".join(list(focal_proxim.nlargest(5).index))
        df_dict = {'Token': name, 'Prom_Node': top_node, 'Nr_ProxNodes': len(focal_proxim), 'NrNodes': len(cluster)}
        df_dict.update(cluster_measures)
        dataframe_list.append(df_dict.copy())

    ego_clusters = pd.DataFrame(dataframe_list)
    ego_clusters = ego_clusters.sort_values(by="w_Avg", ascending=False)
    ego_cluster_ids = list(ego_clusters.index)

    logging.info("Populating year-on-year ego networks.")

    ego_graph = nx.DiGraph()
    for year in ego_years:
        snw.decondition()

        if moving_average is not None:
            start_year = max(ego_years[0], year - moving_average[0])
            end_year = min(ego_years[-1], year + moving_average[1])
            ma_years = np.arange(start_year, end_year + 1)
        else:
            ma_years = year
        logging.info(
            "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                year,
                moving_average[
                    0],
                moving_average[
                    1], ma_years))
        snw.condition(times=ma_years, tokens=ego_tokens, depth=1, weight_cutoff=ego_cutoff,
                      compositional=compositional)
        if symmetric: snw.to_symmetric()
        if sparsity_percentage is not None:
            snw.sparsify(sparsity_percentage)
        ego_proxim_np = snw.pd_format(
            snw.proximities(focal_tokens=ego_tokens, alter_subset=ego_tokens),
            ids_to_tokens=True)[0]
        #ego_proxim_np = cutoff_percentage(ego_proxim_np, sparsity_percentage)
        for i, j in product(ego_cluster_ids, ego_cluster_ids):
            if i != j:
                if cluster_reduction == "max":
                    weight = ego_proxim_np.reindex(replacement_cluster_list[i], axis=0, fill_value=0).reindex(
                        replacement_cluster_list[j],
                        axis=1,
                        fill_value=0).max().max() * scaling_factor
                    if not symmetric:
                        r_weight = ego_proxim_np.reindex(replacement_cluster_list[j], axis=0, fill_value=0).reindex(
                            replacement_cluster_list[i],
                            axis=1,
                            fill_value=0).max().max() * scaling_factor
                else:
                    weight = ego_proxim_np.reindex(replacement_cluster_list[i], axis=0, fill_value=0).reindex(
                        replacement_cluster_list[j],
                        axis=1,
                        fill_value=0).mean().mean() * scaling_factor
                    if not symmetric:
                        r_weight = ego_proxim_np.reindex(replacement_cluster_list[j], axis=0, fill_value=0).reindex(
                            replacement_cluster_list[i],
                            axis=1,
                            fill_value=0).mean().mean() * scaling_factor

                i_token = str(year) + "_" + ego_clusters.loc[[i]].Prom_Node.values[0]
                j_token = str(year) + "_" + ego_clusters.loc[[j]].Prom_Node.values[0]

                i_desc = ego_clusters.loc[[i]].Token.values[0]
                j_desc = ego_clusters.loc[[j]].Token.values[0]

                name_dict = {}
                name_dict[i_token] = {"desc": i_desc}
                name_dict[j_token] = {"desc": j_desc}
                ego_graph.add_nodes_from([i_token, j_token])
                nx.set_node_attributes(ego_graph, name_dict)
                if i_token != j_token:
                    edge = (i_token, j_token, {'weight': weight, 'edgetype': "ego"})
                    if weight > 0:
                        ego_graph.add_edges_from([edge])
                    if not symmetric:
                        edge = (j_token, i_token, {'weight': r_weight, 'edgetype': "ego"})
                        if r_weight > 0:
                            ego_graph.add_edges_from([edge])
        # Add year parameter to node in ego graph
        name_dict = {}
        for node in ego_tokens:
            c_node = str(year) + "_" + node
            name_dict[c_node] = {"token": node, "year": str(year)}
        nx.set_node_attributes(ego_graph, name_dict)

    context_ego_graph = nx.DiGraph()
    context_ego_graph.add_nodes_from(ego_graph.nodes(data=True))
    context_ego_graph.add_edges_from(ego_graph.edges(data=True))
    context_ego_graph.add_nodes_from(context_graph.nodes(data=True))
    context_ego_graph.add_edges_from(context_graph.edges(data=True))

    logging.info("Calculating ties between context and ego network")
    for year in ego_years:
        logging.info("Conditioning context on year {}.".format(year))
        # Ego to context
        snw.decondition()

        if moving_average is not None:
            start_year = max(ego_years[0], year - moving_average[0])
            end_year = min(ego_years[-1], year + moving_average[1])
            ma_years = np.arange(start_year, end_year + 1)
        else:
            ma_years = year
        logging.info(
            "Calculating proximities for fixed relevant clusters for year {} with moving average -{} to {} over {}".format(
                year,
                moving_average[
                    0],
                moving_average[
                    1], ma_years))

        snw.context_condition(times=ma_years, tokens=ego_tokens + context_tokens, depth=1, weight_cutoff=context_cutoff)
        if sparsity_percentage is not None:
            snw.sparsify(sparsity_percentage)
        if symmetric: snw.to_symmetric()

        for i, j in product(ego_cluster_ids, context_clusters_ids):

            ego_subset = replacement_cluster_list[i]
            context_subset = context_cluster_list[j]
            ego_subset_name = str(year) + "_" + ego_clusters.loc[[i]].Prom_Node.values[0]
            context_subset_name = "c_" + context_clusters.loc[[j]].Prom_Node.values[0]

            proxim_np = snw.pd_format(
                snw.proximities(focal_tokens=ego_subset + context_subset, alter_subset=ego_subset + context_subset),
                ids_to_tokens=True)[0]
            ego_to_context = proxim_np.reindex(ego_subset, fill_value=0).reindex(context_subset, fill_value=0, axis=1)
            context_to_ego = proxim_np.reindex(context_subset, fill_value=0).reindex(ego_subset, fill_value=0, axis=1)
            if cluster_reduction == "max":
                weight = ego_to_context.max().max() * scaling_factor
                if symmetric:
                    r_weight = context_to_ego.max().max() * scaling_factor
            else:
                weight = ego_to_context.mean().mean() * scaling_factor
                if symmetric:
                    r_weight = context_to_ego.mean().mean() * scaling_factor
            if ego_subset_name != context_subset_name:
                edge = (ego_subset_name, context_subset_name, {'weight': weight, 'edgetype': "egocontext"})
                if weight > 0:
                    context_ego_graph.add_edges_from([edge])
                if not symmetric:
                    edge = (context_subset_name, ego_subset_name, {'weight': r_weight, 'edgetype': "egocontext"})
                    if r_weight > 0:
                        context_ego_graph.add_edges_from([edge])
                    context_ego_graph.add_edges_from([edge])

    # Set hierarchy
    hierarchy_dict = {}
    for node in context_ego_graph.nodes():
        if node in context_graph.nodes():
            hierarchy_dict[node] = {"hierarchy": 0}
        else:
            hierarchy_dict[node] = {"hierarchy": 1}
    nx.set_node_attributes(context_ego_graph, hierarchy_dict)

    edge_weights = nx.get_edge_attributes(context_ego_graph, 'weight')
    context_ego_graph.remove_edges_from((e for e, w in edge_weights.items() if w <= 0))
    isolates = list(nx.isolates(context_ego_graph))
    logging.debug(
        "Found {} isolated nodes in graph, deleting.".format(len(isolates)))
    context_ego_graph.remove_nodes_from(isolates)

    return context_ego_graph
