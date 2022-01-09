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
setup_logger(config['Paths']['log'], config['General']['logging_level'], "7_embedding.py")

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
keep_top_k = [50, 100, 200, 1000][1]
max_degree = [50, 100][0]
level = 2#[15, 10, 8, 6, 4, 2][1]
keep_only_tokens = [True, False][0]
contextual_relations = [True, False][0]

# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

imagemethod="imshow"
#imagemethod="contour"

sel_alter="boss"
use_diff=True
im_int_method="gaussian"
grid_method="linear"
npts = 200
int_level=8
ngridx = 12
ngridy = ngridx
top_n=2
nr_tokens=500000000
#nr_tokens=50
#nr_tokens=5



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

from karateclub.node_embedding.neighbourhood.geometriclaplacianeigenmaps import GLEE
nodes=list(X.index)
ids = list(range(0,len(X.index)))
index_dict=dict(zip(ids,nodes))
node_dict=dict(zip(nodes,ids))
X.index=ids
X.columns=ids


G = nx.from_pandas_adjacency(X)

kernel_pca = KernelPCA(
    n_components=2, kernel="precomputed",random_state=100)

model = GLEE(dimensions=1,seed=100)
#model.fit(G)
#Y=model.get_embedding()
Y = kernel_pca.fit_transform(X)
pos=Y.copy()
#Y=(y-np.min(y, axis=0, keepdims=True))/(np.max(y,axis=0, keepdims=True)-np.min(y,axis=0, keepdims=True))
G.remove_edges_from(list(nx.selfloop_edges(G)))
G.remove_edges_from(list(nx.selfloop_edges(G)))
G.remove_edges_from(list(nx.selfloop_edges(G)))

Y = pd.DataFrame(Y)
Y.index=nodes
Y["color"] = [item/np.max(color)for item in color]
Y.columns=["x","y","color"]

fig = plt.figure(figsize=(12, 8))
fig.suptitle(
    "{}-{}: Contextual clusters leader vs {}, lvl {}, selected with {}, normed {}, imagefilter: {}".format(years[0],years[-1],sel_alter,level, tf, use_diff,im_int_method), fontsize=14
)
lim_max=np.max(np.max(Y.loc[:,["x","y"]]))
lim_min=np.min(np.min(Y.loc[:,["x","y"]]))
ax = fig.add_subplot(xlim=(lim_min,lim_max), ylim=(lim_min,lim_max))

ax.scatter(Y["x"], Y["y"], s=100, c=Y.color,  cmap="Pastel1")
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > X.mean().mean()]
nx.draw_networkx_edges(G,pos, edgelist=elarge, ax=ax, alpha=0.05)
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


for idx, row in Y.iterrows():
    #ax.annotate(row.name, (row[0], row[1]), xytext=(-25, 4), textcoords='offset points', alpha=0.3)
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
            alpha=0.2
        else:
            alpha=0.5
        ax.annotate(name, (row[0], row[1]), xytext=(5, start-i*d), textcoords='offset points', alpha=alpha)


#for idx, row in Y2.iloc[0:50,:].iterrows():
#    ax.annotate(row.name, (row[0], row[1]), xytext=(-10, 0), textcoords='offset points', alpha=0.6)



# force matplotlib to draw the graph
#ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
ax.axis("tight")

plt.show()
