import logging
import os
import pickle
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
from sklearn.manifold import spectral_embedding

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
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.tri as tri
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

imagemethod="imshow"
#imagemethod="contour"

sel_alter="boss"
use_diff=False
im_int_method="gaussian"
grid_method="linear"
npts = 200
int_level=16
ngridx = 12
ngridy = ngridx
top_n=3
nr_tokens=500000000
#nr_tokens=50
#nr_tokens=5


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
df_clusters=pd.read_excel(checkname)
cldict=pickle.load(open(pname, "rb"))
X=df_clusters.iloc[:,1:-7]
X=X.div(X.sum(axis=1), axis=0)
color=X.index.to_list()
cl_name=df_clusters.iloc[:,0].to_list()
X.index=cl_name

# Get all datapoints
checkname = filename + "REGDF_allY_.xlsx"
df=pd.read_excel(checkname)
X2=df.iloc[:,1:-7]
#X2=X2.div(X2.sum(axis=1), axis=0)
X2["color"] = df.rweight/np.max(df.rweight)
X2["alter"] = semantic_network.ensure_tokens(df.occ)
#summedX2=X2.groupby("alter", as_index=False).sum().sort_values(by="color", ascending=False)
#summedX2.iloc[:,1:-1]=summedX2.iloc[:,1:-1].div(summedX2.iloc[:,1:-1].sum(axis=1), axis=0)
#X2=summedX2
X2=X2.replace([np.inf, -np.inf], np.nan).dropna( how="any")
X2=X2.sort_values(by="color", ascending=False).iloc[0:nr_tokens,:]
X2cols=np.setdiff1d(X2.columns,["alter","color"]).tolist()
# Drop rows with no connections
X2=X2.iloc[np.where(X2[X2cols].sum(axis=1)>0)[0],:]


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
methods["Isomap"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, metric="precomputed")
methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods["SE"] = manifold.SpectralEmbedding(affinity="precomputed",
    n_components=n_components, n_neighbors=n_neighbors
)
methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)
kernel_pca = KernelPCA(
    n_components=2, kernel="precomputed")


import seaborn as sns
# Create figure
fig = plt.figure(figsize=(12, 8))
fig.suptitle(
    "Contextual clusters leader vs {}, lvl {}, selected with {}, normed {}, imagefilter: {}".format(sel_alter,level, tf, use_diff,im_int_method), fontsize=14
)


# Plot results
label="SE"
i=1
method=methods[label]
t0 = time()
#Y = method.fit_transform(X[X2cols])
#Y = spectral_embedding(X[X2cols].to_numpy(), n_components=n_components, )
Y = kernel_pca.fit_transform(X[X2cols])



xx=X2[X2cols].to_numpy()
xsums=np.sum(xx, axis=1,  keepdims=True)
xx=xx/xsums
sorted_row_idx = np.argsort(xx, axis=1)[:,0:-top_n]
col_idx = np.arange(xx.shape[0])[:,None]
xx[col_idx,sorted_row_idx]=0
xx=xx/np.sum(xx, axis=1,  keepdims=True)
Y2 = np.matmul(xx, Y)
print(np.max(Y2, axis=0))
print(np.min(Y2, axis=0))

Y = pd.DataFrame(Y)
Y2 = pd.DataFrame(Y2)
Y.index=X.index
Y["color"] = [item/np.max(color)for item in color]
Y2.index=X2.index
Y2["color"]= X2.loc[:,"color"]
Y2["alter"] = X2["alter"]
Y.columns=["x","y","color"]
Y2.columns=["x","y","color", "alter"]
t1 = time()
print("%s: %.2g sec" % (label, t1 - t0))
ax = fig.add_subplot()


if sel_alter is not None:
    print("Orig Nr of Observations: {}".format(len(Y2)))
    if use_diff:
        Y3=Y2.copy()
    else:
        Y3=None
    Y2=Y2[Y2.alter == sel_alter]
    print("Selected of Observations: {}".format(len(Y2)))

xi = np.linspace(Y.x.min(), Y.x.max(), ngridx)
yi = np.linspace(Y.y.min(), Y.y.max(), ngridy)
extent = (min(Y["x"]), max(Y["x"]), max(Y["y"]), min(Y["y"]))
#xs,ys = np.mgrid[extent[0]:extent[1]:3j, extent[2]:extent[3]:3j]
#resampled = griddata((x, y), z, (xs, ys))

def get_zi(Y2, xi, yi):
    Y2=Y2[~Y2[["x","y"]].duplicated()]
    x=Y2["x"]
    y=Y2["y"]
    z=Y2.color
    xy=Y2[["x","y"]]

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    #interpolator = tri.LinearTriInterpolator(triang, z)
    #Xi, Yi = np.meshgrid(xi, yi)
    #zi = interpolator(Xi, Yi)

    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=grid_method)
    zi[np.isnan(zi)] = 0
    #zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))
    return zi

zi = get_zi(Y2, xi, yi)
if use_diff and sel_alter:
    zi2 = get_zi(Y3, xi, yi)
    zi = zi-zi2


if imagemethod=="contour":
    ax.contour( xi,yi,zi, levels=int_level, linewidths=0.5, colors='k', extent=(min(Y["x"]), max(Y["x"]), max(Y["y"]), min(Y["y"])))
    cntr1 = ax.contourf( xi,yi,zi, levels=int_level, cmap="RdBu_r",extent=(min(Y["x"]), max(Y["x"]), max(Y["y"]), min(Y["y"])))
else:
    cntr1=ax.imshow(zi, extent=(min(Y["x"]), max(Y["x"]), max(Y["y"]), min(Y["y"])), cmap="RdBu_r", origin="lower", interpolation=im_int_method)


fig.colorbar(cntr1, ax=ax)
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
        #ax.annotate(name, (row[0], row[1]), xytext=(5, start-i*d), textcoords='offset points', alpha=alpha)


for idx, row in Y.iterrows():
    ax.annotate(row.name, (row[0], row[1]), xytext=(-10, 0), textcoords='offset points', alpha=0.6)


#for idx, row in Y2.iloc[0:50,:].iterrows():
#    ax.annotate(row.name, (row[0], row[1]), xytext=(-10, 0), textcoords='offset points', alpha=0.6)



# force matplotlib to draw the graph
ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis("tight")

plt.show()
