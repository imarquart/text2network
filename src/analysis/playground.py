# %% Config

import networkx as nx
import numpy as np
import pandas as pd
import time
from config.config import configuration
from src.classes.neo4jnw import neo4j_network
from src.classes.neo4jnw_aggregator import neo4jnw_aggregator

cfg = configuration()

db_uri = "http://localhost:7474"
db_pwd = ('neo4j', 'nlp')
neo_creds = (db_uri, db_pwd)
neograph = neo4j_network(neo_creds)
neoagg = neo4jnw_aggregator(neo_creds)
neograph.condition(1993,"leader")
focal_token = "leader"
year=1993
brange=range(1000,30000,5000)
cutrange=range(0,9,1)
normrange=[True,False]
for batchsize in brange:
    for cut in cutrange:
        cut=cut/10
        for norm in normrange:
            start_time = time.time()
            neograph.condition(year, weight_cutoff=cut, norm=norm, batchsize=batchsize)
            print("BS {}, Cutoff {}, Norm {} - Full conditioning for took {}. {} Nodes, {} Edges".format(batchsize,cut,norm, time.time() - start_time,len(neograph.graph.nodes),len(neograph.graph.edges)))


