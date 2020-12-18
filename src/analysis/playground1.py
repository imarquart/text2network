# %% Config

import networkx as nx
import numpy as np
import pandas as pd
from src.classes.neo4jnw_dummy import neo4j_network_dummy
from src.classes.neo4jnw_measures import neo4jnw_measures
from src.functions.node_mearues import proximities, centrality, yearly_centralities
from src.functions.backout_measure import backout_measure
from src.functions.format import pd_format
from src.utils.network_tools import make_symmetric

neo4nw=neo4j_network_dummy()
print(pd_format(proximities(neo4nw,["president"])))
g2=backout_measure(neo4nw.graph)
print(pd_format(proximities(neo4nw,["president"],nw=g2)))


print(pd_format(centrality(neo4nw,types=['normedPageRank'])))
g2=backout_measure(neo4nw.graph)
print(pd_format(centrality(neo4nw,types=['normedPageRank'],nw=g2)))




g2=make_symmetric(neo4nw.graph)
print(pd_format(proximities(neo4nw,["president"])))
print(pd_format(proximities(neo4nw,["president"],nw=g2)))


print(pd_format(centrality(neo4nw,types=['normedPageRank'])))
print(pd_format(centrality(neo4nw,types=['normedPageRank'],nw=g2)))
print(pd_format(centrality(neo4nw,types=['normedPageRank'])))

neo4nw.graph=g2
print(pd_format(centrality(neo4nw,types=['normedPageRank'],nw=g2)))



test=yearly_centralities(neo4nw,[1991,1992,1993,1994])

#print(pd_format(test))
