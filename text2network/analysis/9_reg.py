import os
import pickle
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import KernelPCA
from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.utils.file_helpers import check_folder
from text2network.utils.logging_helpers import setup_logger
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import statsmodels.api as sm
from statsmodels.formula.api import ols

def get_convex_hull(Y, set_points, starting_point):
    # Leftmost point
    convex_hull = []

    convex_hull.append(Y.loc[starting_point, :])

    ending = False
    i = 0
    while ending == False:
        curpoint = convex_hull[i].copy()
        endpoint = Y.loc[set_points[0], :]
        curset = set_points.copy()
        # curset.remove(curpoint.name)
        for c in curset:
            # print("Checking {}".format(c))
            candidate = Y.loc[c, :]
            Orin = (endpoint.x - curpoint.x) * (candidate.y - curpoint.y) - (candidate.x - curpoint.x) * (
                    endpoint.y - curpoint.y)
            # print(angle2-angle1 )
            if (Orin > 0) or (endpoint.name == curpoint.name):
                endpoint = candidate
        convex_hull.append(endpoint.copy())
        # set_points.remove(endpoint.name)
        if endpoint.name == convex_hull[0].name or i >= len(set_points) + 1:
            ending = True
        else:
            i = i + 1

    return convex_hull


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
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_folder(configuration_path))
config.read(check_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "7_embedding.py")

semantic_network = neo4j_network(config)


# Filename loading
years = list(range(1980, 2021))
focal_token = "leader"

main_folder = "profile_relationships2_bidirectional_ADJ_VERB_NOUN_ADV_ADP"

filename, picdir = get_filename(config['Paths']['csv_outputs'], main_folder,
                                focal_token=focal_token, cutoff=0.1, tfidf="weight",
                                context_mode="bidirectional", contextual_relations=True,
                                postcut=0.01, keep_top_k=1000, depth=0,
                                max_degree=100, algo=consensus_louvain, level=10, rs=100, tf="weight", sub_mode="bidirectional")
picdir = "\\".join(picdir.split("\\")[0:-4])


# %% SEttings
focal_substitutes = focal_token
focal_occurrences = None
cluster_subset = ["business", "people", "team", "good", "organization", "ceo", "better", "company", "make", "global",
                  "work", "help", "industry", "person", "lead", "different", "level", "need", "strong", "leadership"][0:14]

sel_alter2 = None
sel_alter_list = []
sel_alter_list.append(None)
sel_alter_list.append(["ceo", "president", "founder", "successor", "chairman"])
sel_alter_list.append(["manager", "executive", "pioneer", "follower", "champion"])
sel_alter_list.append(["boss", "supervisor", "father", "subordinate", "superior"])

year_list = [1980, 2021]
moving_average = (20, 20)

imagemethod = "imshow"
imagemethod = "contour"
im_int_method = "gaussian"
grid_method = "linear"
npts = 200
int_level = 18
ngridx = 8
ngridy = ngridx
scale_percent = 0.1

# Embedding Settings
top_n = 6
emb_top_n = 30
nr_tokens = 500000000
convex_proximity = False



# %% Get Cluster Data

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes = [focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences = [focal_occurrences]

checkname = filename + "_CLdf" + ".xlsx"
pname = filename + "_CLdict_tk.p"
df_clusters = pd.read_excel(checkname)
cldict = pickle.load(open(pname, "rb"))

# Get Adjacency matrix and drop columns which may be cluster names
X = df_clusters.iloc[:, 1:-7]
X.index = X.columns.to_list()
for dcol in ["year", "type", "pos", "sub", "occ"]:
    if dcol in X.columns:
        X = X.drop(columns=dcol)
    if dcol in X.index:
        X = X.drop(dcol)
cl_name = X.columns.to_list()
assert all(X.columns == X.index)

if cluster_subset is not None:
    X = X[X.index.isin(cluster_subset)]
    X = X.loc[:, X.columns.isin(cluster_subset)]
cl_name = X.index
# X=X.div(X.sum(axis=1), axis=0)
# X.iloc[:,:]=(X.to_numpy()+X.to_numpy().T)/2
xx = X.to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:, 0:-top_n]
col_idx = np.arange(xx.shape[0])[:, None]
xx[col_idx, sorted_row_idx] = 0
X = pd.DataFrame(xx)
# X=X/np.max(X.max())
color = list(range(0, len(X.index)))
X.index = cl_name

if years is None:
    # Get all datapoints
    checkname = filename + "REGDF_allY.xlsx"
    df = pd.read_excel(checkname)
else:
    ylist = []
    for year in years:
        checkname = filename + "REGDF" + str(year) + ".xlsx"
        df = pd.read_excel(checkname)
        df["tYear"] = year
        ylist.append(df.copy())
    df = pd.concat(ylist)

for sel_alter in sel_alter_list:


    if sel_alter is None:
        sel_alter=df.occ.unique().tolist()
    X2 = df[cl_name].copy()
    # X2=X2.div(X2.sum(axis=1), axis=0)
    X2["prob"] = df.rweight / np.max(df.rweight)
    X2["alter"] = semantic_network.ensure_tokens(df.occ)
    X2["year"] = df.tYear
    # summedX2=X2.groupby("alter", as_index=False).sum().sort_values(by="color", ascending=False)
    # summedX2.iloc[:,1:-1]=summedX2.iloc[:,1:-1].div(summedX2.iloc[:,1:-1].sum(axis=1), axis=0)
    # X2=summedX2
    X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    X2 = X2.sort_values(by="prob", ascending=False).iloc[0:nr_tokens, :]
    # Drop rows with no connections
    X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]



    X2.loc[~X2.alter.isin(sel_alter),"prob"]=0
    X2a=X2[X2.alter.isin(sel_alter)]
    X2a=X2
    Y=X2a.prob
    X=X2a[cl_name]
    cl_name_f=list(cl_name)
    cl_name_f[np.where(np.array(cl_name)=="global")[0][0]]="gglobal"
    X.columns=cl_name_f
    X=X/40
    X=X/100
    X.div(Y, axis=0)
    #X=X.div(X.sum(axis=1), axis=0)
    #X=(X-X.mean())/X.std()
    Y=(Y-Y.min())/(Y.max()-Y.min())
    X=sm.add_constant(X)
    X["P2"]=np.int64((X2a.year>2009).to_numpy())
    X["P1"]=np.int64((X2a.year>2000).to_numpy())
    X["year"]=X2a.year
    X["y"]=Y
    f1="y~"+"+".join([x for x in cl_name_f])+"+year+"+"*year+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+C(year)"+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+year+"+"*year+".join([x for x in cl_name_f])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])

    f1="y~"+"+".join([x for x in cl_name_f])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])

    f1="y~"+"+".join([x for x in cl_name_f])+"+year+"+"*year+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+C(year)"+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])
    #f1="y~"+"+".join([x for x in cl_name_f])+"+year+"+"*year+".join([x for x in cl_name_f])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])


    f1="y~"+"+".join([x for x in cl_name_f])+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+C(year)+"+"+C(P2)+"+"*C(P2)+".join([x for x in cl_name_f])
    f1="y~"+"+".join([x for x in cl_name_f])+"+year+year^2+year^3+year^4"+"+C(P1)"+"+C(P2)+"+"*C(P1)+".join([x for x in cl_name_f])+"+"+"*C(P2)+".join([x for x in cl_name_f])

    model = ols(formula=f1, data=X)
    results = model.fit(cov_type='HC1',)
    #print(results.summary())
    results_summary = results.summary()
    tab=results_summary.tables[1]
    print(results.rsquared)

    LRresult = (results.summary2().tables[1])
    sigres=LRresult[LRresult.loc[:,"P>|z|"]<0.05]
    print(sel_alter)
    print(sigres)

