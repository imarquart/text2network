# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:44:13 2020

@author: marquart
"""
class aggregator():
    
       def __init__(self, neo4nw, weight_cutoff=None, norm_ties=False, logging_level=logging.NOTSET):
        # Set logging level
        logging.disable(logging_level)
        self.logging_level=logging_level
        # Create network
        self.neo4nw = neo4nw
        self.norm_ties = norm_ties
        self.weight_cutoff = weight_cutoff
