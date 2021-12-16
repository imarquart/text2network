import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from text2network.classes.neo4jnw import neo4j_network
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = 'config/analyses/FounderSenBert40.ini'
# Settings
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "f_semantic_importance.py")

output_path = check_create_folder(config['Paths']['csv_outputs'])
output_path = check_create_folder(config['Paths']['csv_outputs'] + "/regression_tables/")
filename = check_create_folder("".join([output_path, "/founder_reg"]))

semantic_network = neo4j_network(config)
times = list(range(1980, 2021))

df= pd.read_excel(filename + "REGDF_allY_" + ".xlsx")
df=df.iloc[:,1:]
columns_occ = df.columns[-8:].to_list()
columns_firms = df.columns[:-8].to_list()

df=df[df.loc[:,columns_firms].any(axis=1)]

from sklearn.ensemble import RandomForestRegressor
from rfcc_local.rfcc.model import cluster_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit_transform(y)
X=df.loc[:, columns_firms+["sentiment"]]
y=df.loc[:,["f_w"]]
xsel=X.sum(axis=1)>0.2
X=X.loc[xsel,:]
y=y[xsel]

clf = RandomForestRegressor(n_estimators=100, verbose=2, n_jobs=-1)
clf = clf.fit(X, y)
print(clf.score(X,y))



model=cluster_model(model=RandomForestRegressor,max_clusters=16,random_state=1,n_estimators=50, verbose=2, n_jobs=-1)
model.fit(X,y)

clusters=model.cluster_descriptions(continuous_measures="mean")


