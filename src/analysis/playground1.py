# %% Config

import networkx as nx
import numpy as np
import pandas as pd
from src.classes.neo4jnw_dummy import neo4j_network_dummy
from src.functions.node_measures import proximity, centrality, yearly_centrality
from src.functions.backout_measure import backout_measure
from src.functions.format import pd_format
from src.utils.network_tools import make_symmetric

neo4nw=neo4j_network_dummy()
# neo4nw.conditioned=False
# print(pd_format(neo4nw.proximities()))
# neo4nw.to_backout()
# print(pd_format(neo4nw.proximities()))

# neo4nw=neo4j_network_dummy()
# neo4nw.conditioned=False
# print(pd_format(neo4nw.centrality()))
# neo4nw.to_backout()
# print(pd_format(neo4nw.centrality()))

neo4nw=neo4j_network_dummy()
print(pd_format(neo4nw.proximities(focal_tokens=['president','tyrant'],alter_subset=['man', 'woman'])))
neo4nw.to_backout()
print(pd_format(neo4nw.proximities(focal_tokens=['president','tyrant'],alter_subset=['man', 'woman'])))
neo4nw=neo4j_network_dummy()
neo4nw.to_symmetric()
print(pd_format(neo4nw.proximities(focal_tokens=['president','tyrant'],alter_subset=['man', 'woman'])))



neo4nw=neo4j_network_dummy()
print(pd_format(neo4nw.centralities(focal_tokens=['president','tyrant','man', 'woman'])))
neo4nw.to_backout()
print(pd_format(neo4nw.centralities(focal_tokens=['president','tyrant','man', 'woman'])))
neo4nw=neo4j_network_dummy()
neo4nw.to_symmetric()
print(pd_format(neo4nw.centralities(focal_tokens=['president','tyrant','man', 'woman'])))


measures=[neo4nw.proximities(focal_tokens=['president','tyrant'],alter_subset=['man', 'woman']),neo4nw.centralities(focal_tokens=['president','tyrant','man', 'woman'])]