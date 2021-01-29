# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:16:52 2020

@author: marquart
"""
from _collections import defaultdict
import logging
from typing import Optional, Callable, Tuple, List, Dict, Union

from src.utils.network_tools import make_symmetric
import networkx as nx
from community import best_partition

# Type definition
try:
    from typing import TypedDict
    class GraphDict(TypedDict):
        graph: nx.DiGraph
        name: str
        parent: int
        level: int
        measures: List
        metadata: [Union[Dict,defaultdict]]
except:
    GraphDict = Dict[str, Union[str,int,Dict,List,defaultdict]]

def louvain_cluster(graph):
    graph = make_symmetric(graph)
    clustering = best_partition(graph)

    return [[k for k, v in clustering.items() if v == val] for val in list(set(clustering.values()))]



def cluster_graph(graph: GraphDict, to_measure: Optional[List[Callable[[nx.DiGraph], Dict]]] = None,
                  algorithm: Optional[Callable[[nx.DiGraph], List]] = None) -> Tuple[GraphDict, List[GraphDict]]:
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

    Returns
    -------
    Modified original graph, list of subgraphs
    """

    subgraph_list = []
    # Extract information
    graph_name = graph['name']
    graph_level = graph['level']
    graph_metadata = graph['metadata']

    if algorithm==None:
        algorithm=louvain_cluster

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

    # Add cluster attribute to original graph
    nx.set_node_attributes(graph['graph'], cluster_node_dict, 'cluster')

    # Apply measures to original graph
    measure_list = []
    if to_measure is not None:
        for measure in to_measure:
            measure_list.append(measure(nw_graph=graph['graph']))
    graph['measures']=measure_list


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

    return graph, subgraph_list


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
    if copy == True:
        cluster_subgraph = nx.subgraph(graph, nodelist).copy()
    else:
        cluster_subgraph = nx.subgraph(graph, nodelist)

    return cluster_subgraph


def return_cluster(graph: nx.DiGraph, name: str, parent: str, level: int, measures: Optional[List] = None,
                   metadata: Optional[Union[dict,defaultdict]] = None) -> GraphDict:
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
    measure_dicts =list()
    if metadata is not None:
        for (k,v) in metadata.items():
            metadata_dict[k].append(v)
    if measures is not None:
        for mdict in measures:
            measure_dicts.append(mdict)
    graph_dict = {'graph': graph, 'name': name, 'parent': parent, 'level': level, 'measures': measure_dicts,
                  'metadata': metadata_dict}

    return graph_dict