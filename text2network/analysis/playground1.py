# %% Config

import networkx as nx
import numpy as np
import pandas as pd
from src.classes.neo4jnw_dummy import neo4j_network_dummy
from src.functions.node_measures import proximity, centrality, yearly_centrality
from src.functions.backout_measure import backout_measure
from src.functions.format import pd_format
from src.utils.network_tools import make_symmetric
from src.functions.graph_clustering import *

# neo4nw.conditioned=False
# print(pd_format(neo4nw.proximities()))
# neo4nw.to_backout()
# print(pd_format(neo4nw.proximities()))

# neo4nw=neo4j_network_dummy()
# neo4nw.conditioned=False
# print(pd_format(neo4nw.centrality()))
# neo4nw.to_backout()
# print(pd_format(neo4nw.centrality()))

#measures=[neo4nw.proximities(focal_tokens=['president','tyrant'],alter_subset=['man', 'woman']),neo4nw.centralities(focal_tokens=['president','tyrant','man', 'woman'])]



neo4nw=neo4j_network_dummy()
metadata={"year":2000,"desc":"Test"}
measures=[]

graphcluster=return_cluster(neo4nw.graph,"Test","",0,measures,metadata)

neo4nw.condition(years=[1992,2005], weight_cutoff=0.05, context=['USA','China'])
graphcluster=return_cluster(neo4nw.graph,name="Test",parent="",level=0,measures=[],metadata={'years':[1992,2005], 'context':['USA','China']})
base_cluster,clusters=cluster_graph(graphcluster, to_measure=[proximity, centrality],algorithm=louvain_cluster)
for cl in clusters:
    print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'], cl['level'], cl['parent'], cl['graph'].nodes))
    print(pd_format(cl['measures']))

#asdf=cluster_graph(graphcluster, to_measure=[proximity, centrality],algorithm=louvain_cluster)
#print(asdf[0])
#print(asdf[1])
#print(asdf[1][1]['name'])
#print(asdf[1][1]['graph'].nodes(data=True))
#print(asdf[1][1]['measures'])
#print(pd_format(asdf[1][1]['measures']))

clusters=neo4nw.cluster(levels=1, to_measure=[proximity,centrality])
print(pd_format(clusters[0]['measures']))

levels=2
clusters=neo4nw.cluster(levels=levels)

for cl in clusters:
    print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'],cl['level'],cl['parent'],cl['graph'].nodes))