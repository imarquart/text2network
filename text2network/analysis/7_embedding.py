import logging
import os
import pickle

import pandas as pd
import numpy as np
from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.utils.file_helpers import check_folder
from text2network.utils.logging_helpers import setup_logger

from collections import OrderedDict
from functools import partial
from time import time
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "7_embedding.py")

semantic_network = neo4j_network(config)


times = list(range(1980, 2021))
focal_token = "leader"
sym = False
rev = False
rs = [100][0]
cutoff = [0.2, 0.1, 0.01][0]
postcut = [0.01, None][0]
depth = [0, 1][0]
context_mode = ["bidirectional", "substitution", "occurring"][0]
algo = consensus_louvain
pos_list = ["NOUN", "ADJ", "VERB"]
tf = ["weight", "diffw", "pmi_weight"][-2]
keep_top_k = [50, 100, 200, 1000][0]
max_degree = [50, 100][0]
level = [15, 10, 8, 6, 4, 2][0]
keep_only_tokens = [True, False][0]
contextual_relations = [True, False][0]

# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes = [focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences = [focal_occurrences]
output_path = check_folder(config['Paths']['csv_outputs'] + "/profile_relationships2/")
output_path = check_folder(
    "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
             str(tf is not None), "_cm", str(context_mode), "/", "conRel", str(contextual_relations), "_postcut",
             str(int(postcut * 100)), "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "/", "lev",
             str(level), "/"]))
logging.info("Getting tf-idf: {}".format(tf))
filename = check_folder("".join([output_path, "/" + "tf_", str(tf) + "/"]))
filename = "".join(
    [filename, "/md", str(max_degree), "_algo", str(algo.__name__),
     "_rs", str(rs)])
logging.info("Overall Profiles  Regression tables: {}".format(filename))
checkname = filename + "_CLdf" + ".xlsx"
pname = filename + "_CLdict_tk.p"
df=pd.read_excel(checkname)
cldict=pickle.load(open(pname, "rb"))
X=df.iloc[:,1:-7]
X=X.div(X.sum(axis=1), axis=0)
color=X.index.to_list()
cl_name=df.iloc[:,0].to_list()
X.index=cl_name

n_neighbors = len(X)-1
n_components = 2

# Set-up manifold methods
LLE = partial(
    manifold.LocallyLinearEmbedding,
    n_neighbors=n_neighbors,
    n_components=n_components,
    eigen_solver="auto",
)

methods = OrderedDict()
methods["LLE"] = LLE(method="standard", n_jobs=-1)
methods["LTSA"] = LLE(method="ltsa", n_jobs=-1)
methods["Hessian LLE"] = LLE(method="hessian", eigen_solver='dense', n_jobs=-1)
methods["Modified LLE"] = LLE(method="modified", n_jobs=-1)
methods["Isomap"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods["SE"] = manifold.SpectralEmbedding(
    n_components=n_components, n_neighbors=n_neighbors
)
methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)



import seaborn as sns
# Create figure
fig = plt.figure(figsize=(8, 8))
fig.suptitle(
    "Contextual clusters of leader, at level {}, selected with {}".format(level, tf), fontsize=14
)


# Get all datapoints
checkname = filename + "REGDF_allY_.xlsx"
df=pd.read_excel(checkname)
X2=df.iloc[:,1:-7]
X2=X2.div(X2.sum(axis=1), axis=0)
X2["color"] = df.rweight/np.max(df.rweight)
X2=X2.replace([np.inf, -np.inf], np.nan).dropna( how="any")



# Plot results
label="Hessian LLE"
i=1
method=methods[label]
t0 = time()
Y = method.fit_transform(X)
Y2 = method.transform(X2.iloc[:,0:-1])

Y = pd.DataFrame(Y)
Y.index=X.index
Y["color"] = [item/np.max(color)for item in color]
Y.columns=["x","y","color"]
t1 = time()
print("%s: %.2g sec" % (label, t1 - t0))
ax = fig.add_subplot()
ax.scatter(Y2[:,0], Y2[:,1], s=20, alpha=0.05, c=X2.color, cmap="afmhot_r")
ax.scatter(Y["x"], Y["y"], s=100, c=Y.color,  cmap="Pastel1")

# Identify extremal points
extremal_points=[]
extremal_points.append(Y.iloc[np.argmax(Y.x),:])
extremal_points.append(Y.iloc[np.argmax(Y.y),:])
extremal_points.append(Y.iloc[np.argmin(Y.x),:])
extremal_points.append(Y.iloc[np.argmin(Y.y),:])

for idx, row in enumerate(extremal_points):
    names=cldict[row.name]
    names.remove(row.name)
    names.insert(0, row.name)
    names=names#[0:4]
    n = len(names)
    d= 8
    dist= n*d
    start=dist//2
    for i,name in enumerate(names):
        if i>0:
            alpha=0.5
        else:
            alpha=1
        ax.annotate(name, (row[0], row[1]), xytext=(5, start-i*d), textcoords='offset points', alpha=alpha)



# force matplotlib to draw the graph
ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis("tight")

plt.show()
