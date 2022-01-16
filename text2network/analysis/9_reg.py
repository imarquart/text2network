import logging
import os
import pickle

from matplotlib.image import NonUniformImage
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
from sklearn.manifold import spectral_embedding

from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.utils.file_helpers import check_folder, check_create_folder
from text2network.utils.logging_helpers import setup_logger

from collections import OrderedDict
from functools import partial
from time import time
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.tri as tri
import networkx as nx

def get_filename(csv_path, main_folder, focal_token, cutoff, tfidf, context_mode, contextual_relations,
                 postcut, keep_top_k, depth, max_degree=None, algo=None, level=None, rs=None, tf=None, sub_mode=None):
    output_path = check_folder(csv_path)
    output_path = check_folder(csv_path + "/" + main_folder + "/")
    output_path = check_folder(
        "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
                 str(tfidf is not None), "_cm", str(context_mode), "/"]))
    output_path = check_folder("".join(
        [output_path, "/", "conRel", str(contextual_relations), "_postcut", str(int(postcut * 100)), "/"]))
    output_path = check_folder("".join(
        [output_path, "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "/"]))
    if max_degree is not None and algo is not None:
        output_path = check_folder("".join(
            [output_path, "/", "md", str(max_degree), "_algo", str(algo.__name__), "/"]))
    if sub_mode is not None:
        output_path = check_folder("".join(
            [output_path, "/", "submode", str(sub_mode), "/"]))
    if level is not None:
        output_path = check_folder("".join(
            [output_path, "/", "lev", str(level), "/"]))
    if tf is not None:
        output_path = check_folder("".join([output_path, "/" + "tf_", str(tf) + "/"]))
    filename = "".join(
        [output_path, "/",
         "rs", str(rs)])
    return filename, output_path




# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Settings

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_folder(configuration_path))
config.read(check_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "9_reg.py")

semantic_network = neo4j_network(config)



years=list(range(1980,2021))
focal_token = "leader"
sym = False
rev = False
rs = [100][0]
cutoff = [0.2, 0.1, 0.01][0]
postcut = [0.2,0.1,0.01, None][0]
depth = [0, 1][0]
context_mode = ["bidirectional", "substitution", "occurring"][1]
sub_mode = ["bidirectional","occurring", "substitution", ][2]#"bidirectional"
algo = consensus_louvain
pos_list = ["NOUN", "ADJ", "VERB"]
tf = ["weight", "diffw", "pmi_weight"][2]
keep_top_k = [50, 100, 200, 1000][2]
max_degree = [50, 100][0]
level = 2#[15, 10, 8, 6, 4, 2][1]
keep_only_tokens = [True, False][0]
contextual_relations = [True, False][0]

# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

sel_alter=["boss","supervisor","father","subordinate","superior"]
sel_alter=["manager","executive","pioneer","follower","champion"]

top_n=50000
nr_tokens=500000000



years=list(range(2015,2021))

#

#
#years=list(range(1992,2005))

#years=list(range(1992,2005))
#years=list(range(1980,2021))
years=list(range(1992,2005))
#years=list(range(1980,2007))
years=list(range(2005,2021))
#years=list(range(1980,1993))
years=list(range(1980,2021))
filename, load_output_path = get_filename(config['Paths']['csv_outputs'], "profile_relationships_sub",
                                     focal_token=focal_token, cutoff=cutoff, tfidf=tf,
                                     context_mode=context_mode, contextual_relations=contextual_relations,
                                     postcut=postcut, keep_top_k=keep_top_k, depth=depth,
                                     max_degree=max_degree, algo=algo, level=level, rs=rs, tf=tf, sub_mode=sub_mode)

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes = [focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences = [focal_occurrences]


logging.info("Overall Profiles  Regression tables: {}".format(filename))
checkname = filename + "_CLdf" + ".xlsx"
pname = filename + "_CLdict_tk.p"
df_clusters=pd.read_excel(checkname)
cldict=pickle.load(open(pname, "rb"))
#df_clusters=df_clusters.drop(columns="type")
X=df_clusters.iloc[:,1:-7]
X=X.div(X.sum(axis=1), axis=0)
xx=X.to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:,0:-top_n]
col_idx = np.arange(xx.shape[0])[:,None]
xx[col_idx,sorted_row_idx]=0
X.iloc[:,:]=(X.to_numpy()+X.to_numpy().T)/2
#X=X/np.max(X.max())
color=X.index.to_list()
cl_name=df_clusters.iloc[:,0].to_list()
X.index=cl_name
Xcols=list(X.columns)

if years is None:
    # Get all datapoints
    checkname = filename + "REGDF_allY.xlsx"
    df=pd.read_excel(checkname)
else:
    ylist = []
    for year in years:
        checkname = filename + "REGDF" + str(year) + ".xlsx"
        df = pd.read_excel(checkname)
        ylist.append(df.copy())
    df= pd.concat(ylist)
X2=df.iloc[:,1:-7]
X2=X2[Xcols]
#X2=X2.div(X2.sum(axis=1), axis=0)
X2["prob"] = df.rweight#/np.max(df.rweight)
X2["alter"] = semantic_network.ensure_tokens(df.occ)
X2["year"] = df.year
#summedX2=X2.groupby("alter", as_index=False).sum().sort_values(by="color", ascending=False)
#summedX2.iloc[:,1:-1]=summedX2.iloc[:,1:-1].div(summedX2.iloc[:,1:-1].sum(axis=1), axis=0)
#X2=summedX2
X2=X2.replace([np.inf, -np.inf], np.nan).dropna( how="any")
X2=X2.sort_values(by="prob", ascending=False).iloc[0:nr_tokens,:]
# Drop rows with no connections
X2=X2.iloc[np.where(X2[Xcols].sum(axis=1)>0)[0],:]

xx=X2[Xcols].to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:,0:-top_n]
col_idx = np.arange(xx.shape[0])[:,None]
xx[col_idx,sorted_row_idx]=0
#xx=xx/np.sum(xx, axis=1,  keepdims=True)

X2.loc[:,Xcols]=xx
import statsmodels.api as sm
from statsmodels.formula.api import ols

#X2.loc[~X2.alter.isin(sel_alter),"prob"]=0
#X2=X2[X2.alter.isin(sel_alter)]


Y=X2.prob
X=X2.iloc[:,0:-3]
X=X/40
X=X/100
X.div(Y, axis=0)
#X=X.div(X.sum(axis=1), axis=0)
Y[~X2.alter.isin(sel_alter)]=0
#X=(X-X.mean())/X.std()
#Y=(Y-Y.mean())/Y.std()
#X=sm.add_constant(X)
X["P2"]=np.int64((X2.year>2007).to_numpy())
X["P1"]=np.int64((X2.year<1993).to_numpy())
X["year"]=X2.year
X["y"]=Y
f1="y~"+"+".join([x for x in Xcols])+"+C(year)"+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in Xcols])+"+"+"*C(P2)+".join([x for x in Xcols])
f1="y~"+"+".join([x for x in Xcols])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in Xcols])+"+"+"*C(P2)+".join([x for x in Xcols])

model = ols(formula=f1, data=X)
results = model.fit()
print(results.summary())
results_summary = results.summary()
tab=results_summary.tables[1]

LRresult = (results.summary2().tables[1])
sigres=LRresult[LRresult.loc[:,"P>|t|"]<0.05]
print(sigres)