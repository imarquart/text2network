# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 21:16:52 2020

@author: marquart
"""
from _collections import defaultdict


def return_cluster(graph,name,parent,measures=None,metadata=None):
    """
    Returns a dict of a cluster, including relevant metadata and associated measures

    Parameters
    ----------
    graph: networkx graph
    name: string
    parent: string
        Name of parent graph
    measures: list
        List of measure dicts, where first level of dictionary gives the name of the measure
    metadata: list of tuples
        List of metadata tuples of the form (key, value)

    Returns
    -------
    Parameterized dict, including two defaultdicts for measures and metadata
    """
    # Package metadata and measures into default dicts
    metadata_dict=defaultdict(list)
    measure_dict=defaultdict(list)
    for k,v in metadata:
        metadata_dict[k].append(v)
    for mdict in measures:
        for k in list(mdict.keys()):
            measure_dict[k].append(mdict[k])

    return {'graph':graph, 'name': name, 'parent': parent,'measures': measure_dict, 'metadata':metadata_dict}