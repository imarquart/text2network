# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:16:52 2020

@author: marquart
"""
import itertools
import logging
from _collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict, Union

import networkx as nx
from community import best_partition

from text2network.functions.network_tools import make_symmetric

# Type definition
try:
    from typing import TypedDict


    class GraphDict(TypedDict):
        graph: nx.DiGraph
        name: str
        parent: int
        level: int
        measures: List
        metadata: [Union[Dict, defaultdict]]
except:
    GraphDict = Dict[str, Union[str, int, Dict, List, defaultdict]]


def louvain_cluster(graph):
    if graph.is_directed():
        graph = make_symmetric(graph)
    clustering = best_partition(graph)

    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]


def consensus_louvain(graph, iterations=4):
    if graph.is_directed():
        graph = make_symmetric(graph)

    new_graph_list = []

    # Run Clustering several times with different starting points
    for i in range(0, iterations):
        new_graph = nx.Graph()
        new_graph.add_nodes_from(graph.nodes)
        clustering = best_partition(graph, randomize=True)
        clustering = [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]
        # Create ties for nodes that belong to the same cluster
        for cluster in clustering:
            pairs = itertools.combinations(cluster, r=2)
            new_graph.add_edges_from(list(pairs))
        new_graph_list.append(new_graph)

    # Consensus clustering
    consensus_graph = nx.Graph()
    consensus_graph.add_nodes_from(graph.nodes)
    pairs = itertools.combinations(list(consensus_graph.nodes), r=2)
    # Iterate over all pairs and add edges
    for u, v in pairs:
        weight = 0
        # Add 1 for each tie in a cluster graph
        for i, cluster_graph in enumerate(new_graph_list):
            if cluster_graph.has_edge(u, v):
                weight = weight + 1
        # Normalize percentage
        weight = weight / iterations
        # Add if majority agrees on tie
        if weight >= 0.5:
            consensus_graph.add_edge(u, v, weight=weight)
    del new_graph_list
    # Now re-run clustering on consensus graph to get final clustering
    clustering = best_partition(consensus_graph)
    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]


def cluster_graph(graph: GraphDict, to_measure: Optional[List[Callable[[nx.DiGraph], Dict]]] = None,
                  algorithm: Optional[Callable[[nx.DiGraph], List]] = None, add_ego_tokens: Optional[list] = None) -> \
Tuple[GraphDict, List[GraphDict], dict]:
    """
    Cluster a graph in a GraphDict into its subgraphs and create measures

    Parameters
    ----------
    graph: GraphDict
        GraphDict container
    to_measure: list
        Callables taking a nw_graph:nx.DiGraph parameters
    algorithm: function
        Callable taking a nx.DiGraph as first argument and returning a list of list of nodes.
    add_ego_tokens: list of ints. Optional
        Optional list of token ids that will be added to each cluster.

    Returns
    -------
    Modified original graph, list of subgraphs, node assignment dictionaries
    """

    subgraph_list = []
    # Extract information
    graph_name = graph['name']
    graph_level = graph['level']
    graph_metadata = graph['metadata']

    if algorithm is None:
        algorithm = louvain_cluster

    # Run clustering algorithm
    try:
        node_list = algorithm(graph['graph'])
    except:
        msg = "Call of clustering algorithm {} failed".format(algorithm)
        logging.error(msg)
        raise AttributeError(msg)

    # Create dict of nodes->cluster associations
    cluster_node_dict = {}
    for i, nodes in enumerate(node_list):
        for node in nodes:
            cluster_node_dict.update({node: i})

    if add_ego_tokens is not None:  # Add ego tokens to each cluster
        node_list = [list(set(x + add_ego_tokens)) for x in node_list]
        for ego in add_ego_tokens:
            if ego not in list(cluster_node_dict.keys()):
                cluster_node_dict[ego] = 0

    # Add cluster attribute to original graph
    nx.set_node_attributes(graph['graph'], cluster_node_dict, 'cluster')

    # Apply measures to original graph
    measure_list = []
    if to_measure is not None:
        for measure in to_measure:
            measure_list.append(measure(nw_graph=graph['graph']))
    graph['measures'] = measure_list

    for i, nodes in enumerate(node_list):
        cluster_subgraph = create_cluster_subgraph(graph['graph'], nodes)
        name = graph_name + "-" + str(i)
        level = graph_level + 1
        # Run measures
        measure_list = []
        if to_measure is not None:
            for measure in to_measure:
                measure_list.append(measure(nw_graph=cluster_subgraph))

        cluster = return_cluster(cluster_subgraph, name, graph_name, level, measure_list, graph_metadata)
        subgraph_list.append(cluster)

    return graph, subgraph_list, cluster_node_dict


def create_cluster_subgraph(graph: nx.DiGraph, nodelist: list, copy=True) -> nx.DiGraph:
    """
    Create a subgraph from list

    Parameters
    ----------
    graph: networkx graph
        Original graph
    nodelist: list
        List of nodes in graph
    copy: bool
        Whether to return a copy or a view. Default is True.

    Returns
    -------
    networkx graph
    """
    if copy:
        cluster_subgraph = nx.subgraph(graph, nodelist).copy()
    else:
        cluster_subgraph = nx.subgraph(graph, nodelist)

    return cluster_subgraph


def return_cluster(graph: nx.DiGraph, name: str, parent: str, level: int, measures: Optional[List] = None,
                   metadata: Optional[Union[dict, defaultdict]] = None) -> GraphDict:
    """
    Returns a dict of a cluster, including relevant metadata and associated measures

    Parameters
    ----------
    graph: networkx graph
        Clustered subgraph
    name: string
    parent: string
        Name of parent graph
    level: int
        Depth of clustering level
    measures: list
        List of measure dicts, where first level of dictionary gives the name of the measure
    metadata: list of tuples
        List of metadata tuples of the form (key, value)

    Returns
    -------
    Parameterized dict, including two defaultdicts for measures and metadata
    """
    # Package metadata and measures into default dicts
    metadata_dict = defaultdict(list)
    measure_dicts = []
    if metadata is not None:
        for (k, v) in metadata.items():
            metadata_dict[k].append(v)
    if measures is not None:
        for mdict in measures:
            measure_dicts.append(mdict)
    graph_dict = {'graph': graph, 'name': name, 'parent': parent, 'level': level, 'measures': measure_dicts,
                  'metadata': metadata_dict}

    return graph_dict
