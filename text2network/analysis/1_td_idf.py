import pandas as pd
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger
import numpy as np
from text2network.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings
years = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "1_td_idf.py")


import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
# First, create an empty network
semantic_network = neo4j_network(config)


focal_tokens=["leader","manager"]
cutoff=0.1

for focal_token in focal_tokens:
    qry="Match p=(r:edge)<-[:onto]-(v:word {token:'"+focal_token+"'}) WITH DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos MATCH (q:edge {run_index:ridx})<-[:onto]-(x:word) WHERE not q.pos in rpos RETURN DISTINCT(x.token) as idx, sum(q.weight)"
    asdf=pd.DataFrame(semantic_network.db.receive_query(qry))
    asdf.columns=["idx","occ"]
    asdf=asdf.sort_values(by="occ", ascending=False)

    qry2="MATCH p=(r:edge)<-[:onto]-(v:word) WHERE v.token in "+str(list(asdf.idx))+" RETURN DISTINCT(v.token) as idx, sum(r.weight) as occ"
    asdf2=pd.DataFrame(semantic_network.db.receive_query(qry2))
    asdf2.columns=["idx","occ_all"]
    asdf2=asdf2.sort_values(by="occ_all", ascending=False)

    asdf3=pd.merge(left=asdf,right=asdf2, how="inner", on=["idx"])
    asdf3['tdn']=100*asdf3['occ']/asdf3['occ_all']
    asdf3['idf']=np.sum(asdf3['occ_all'])/asdf3['occ_all']
    asdf3['tdidf']=asdf3['occ']*np.log(asdf3['idf'])
    asdf3['ntdidf'] = asdf3['tdn'] * np.log(asdf3['idf'])
    asdf3=asdf3.sort_values(by="tdidf", ascending=False)

    filename = "".join(
        [config['Paths']['csv_outputs'], "/tdidf_", str(focal_token), ".xlsx"])

    filename = check_create_folder(filename)
    asdf3.to_excel(filename, merge_cells=False)