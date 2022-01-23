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
cluster_subset=["team", "instance", "good", "organization", "company", "power", "making", "make", "leaders", "management", "people", "business", "need", "see", "project", "ask", "executive", "level", "different", "leadership",]
cluster_subset=["team", "people", "good", "organization", "company", "power"]
cluster_subset=["team", "good", "organization", "management", "people", "need", "leaders", "company", "business", "making", "project", "ask", "executive", "leadership", "make", "see", "know", "must", "role", "employees", "success", "ceo", "process", "global", "kind", "effective", "power", "might", "best", "board",]
cluster_subset=["team", "good", "organization", "management", "people", "need", "leaders", "company", ]
cluster_subset=["team", "management", "company", "business", "people", "process", "leadership", "leaders", "organization", "support", "project", "executive", "customers", "good", "ceo",]

#cluster_subset = None
years=list(range(1980,2021))
main_folder="profile_relationships_substitution_ADJ_VERB_NOUN_ADV_ADP_done"
focal_token = "leader"
sym = False
rev = False
rs = [100][0]
cutoff = [0.2, 0.1, 0.01][1]
postcut = [0.2,0.1,0.01][-1]
depth = [0, 1][0]
context_mode = ["bidirectional", "substitution", "occurring"][1]
sub_mode = ["bidirectional","occurring", "substitution", ][2]#"bidirectional"
algo = consensus_louvain
pos_list = ["NOUN", "ADJ", "VERB"][0]
tf = ["weight", "diffw", "pmi_weight"][0]
keep_top_k = [50, 100, 200, 1000][-1]
max_degree = [50, 100,500][-1]
level = 5#[15, 10, 8, 6, 4, 2][1]
keep_only_tokens = [True, False][0]
contextual_relations = [True, False][0]


# %% Sub or Occ
focal_substitutes = focal_token
focal_occurrences = None

imagemethod="imshow"
imagemethod="contour"
sel_alter2=None
sel_alter=["manager","executive","pioneer","follower","champion"]
sel_alter=["boss","supervisor","father","subordinate","superior"]
#sel_alter=["ceo","president","founder","successor","chairman"]
#sel_alter=None
use_diff=True
im_int_method="gaussian"
grid_method="linear"
npts = 200
int_level=16
ngridx = 12
ngridy = ngridx
top_n=3
top_n_emb=5
nr_tokens=500000000
max_zi=0.1

if use_diff:
    min_zi=-max_zi
else:
    min_zi=0

#years=list(range(1992,2005))
#years=list(range(2005,2021))
years=list(range(1980,1990))
years=list(range(1980,2021))

filename, load_output_path = get_filename(config['Paths']['csv_outputs'], main_folder,
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
cl_name=df_clusters.iloc[:,0].to_list()
X.index=cl_name
if cluster_subset is not None:
    X=X[X.index.isin(cluster_subset)]
    X=X.loc[:,X.columns.isin(cluster_subset)]
cl_name=X.index
#X=X.div(X.sum(axis=1), axis=0)
#X.iloc[:,:]=(X.to_numpy()+X.to_numpy().T)/2
xx=X.to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:,0:-top_n_emb]
col_idx = np.arange(xx.shape[0])[:,None]
xx[col_idx,sorted_row_idx]=0
X=pd.DataFrame(xx)
#X=X/np.max(X.max())
color=X.index.to_list()
X.index=cl_name
X.columns=X.index
Xcols=list(X.columns)


n_neighbors = len(X)-1
n_components = 2
kernel_pca = KernelPCA(
    n_components=2, kernel="precomputed",random_state=100)
# Plot results
label="SE"
i=1
t0 = time()
y = kernel_pca.fit_transform(X)


Y=(y-np.min(y, axis=0, keepdims=True))/(np.max(y,axis=0, keepdims=True)-np.min(y,axis=0, keepdims=True))
Y = pd.DataFrame(Y)
Y.index=X.index
Y["color"] = [item/np.max(color)for item in color]
Y.columns=["x","y","color"]




if years is None:
    # Get all datapoints
    checkname = filename + "REGDF_allY.xlsx"
    df=pd.read_excel(checkname)
else:
    ylist = []
    for year in years:
        checkname = filename + "REGDF" + str(year) + ".xlsx"
        df = pd.read_excel(checkname)
        df["tYear"]=year
        ylist.append(df.copy())
    df= pd.concat(ylist)
X2=df.iloc[:,1:-7]
X2=X2[Xcols]
#X2=X2.div(X2.sum(axis=1), axis=0)
X2["color"] = df.rweight/np.max(df.rweight)
X2["alter"] = semantic_network.ensure_tokens(df.occ)
X2["year"] = df.tYear
#summedX2=X2.groupby("alter", as_index=False).sum().sort_values(by="color", ascending=False)
#summedX2.iloc[:,1:-1]=summedX2.iloc[:,1:-1].div(summedX2.iloc[:,1:-1].sum(axis=1), axis=0)
#X2=summedX2
X2=X2.replace([np.inf, -np.inf], np.nan).dropna( how="any")
X2=X2.sort_values(by="color", ascending=False).iloc[0:nr_tokens,:]
# Drop rows with no connections
X2=X2.iloc[np.where(X2[Xcols].sum(axis=1)>0)[0],:]

if sel_alter is not None:
    XYear=X2[X2.alter.isin(sel_alter)].groupby("year")[Xcols].std().sum(axis=1)
    plt.plot(XYear)
    plt.title("Within STD {}".format(",".join(sel_alter)))
else:
    XYear=X2.groupby("year")[Xcols].std().sum(axis=1)
    plt.plot(XYear)
plt.show()

if sel_alter is not None:
    XYear = np.abs(X2[X2.alter.isin(sel_alter)].groupby("year")[Xcols].mean().diff(1).sum(axis=1))
    plt.plot(XYear)
    plt.title("Between STD {}".format(",".join(sel_alter)))
else:
    XYear=np.abs(X2.groupby("year")[Xcols].mean().diff(1).sum(axis=1))
    plt.plot(XYear)
plt.show()


xx=X2[Xcols].to_numpy()
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
Y2["year"] = X2.year
Y.columns=["x","y","color"]
Y2.columns=["x","y","color", "alter", "year"]
t1 = time()

Y2all = Y2.copy()

year_list=[1981,2006]
moving_average=(1,20)
for year in year_list:
    start_year = max(years[0], year - moving_average[0])
    end_year = min(years[-1], year + moving_average[1])
    ma_years = list(np.arange(start_year, end_year + 1))
    Y2=Y2all[Y2all.year.isin(ma_years)]

    import seaborn as sns
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        "{}-{}: Contextual clusters leader vs {}, lvl {}, selected with {}, normed {}, imagefilter: {}".format(ma_years[0],ma_years[-1],sel_alter,level, tf, use_diff,im_int_method), fontsize=14
    )
    lim_max=np.max(np.max(Y.loc[:,["x","y"]]))
    lim_min=np.min(np.min(Y.loc[:,["x","y"]]))
    ax = fig.add_subplot(xlim=(lim_min,lim_max), ylim=(lim_min,lim_max))

    if sel_alter is not None:
        print("Orig Nr of Observations: {}".format(len(Y2)))
        if use_diff:
            Y3=Y2.copy()
            if sel_alter2 is not None:
                Y3=Y3[Y3.alter.isin(sel_alter2)]
        else:
            Y3=None
        Y2=Y2[Y2.alter.isin(sel_alter)]
        print("Selected of Observations: {}".format(len(Y2)))
    else:
        if use_diff:
            Y3=Y2all.copy()

    xi = np.linspace(lim_min, lim_max, ngridx)
    yi = np.linspace(lim_min, lim_max, ngridx)

    xi = np.linspace(min(Y["x"])-0.01, max(Y["x"])+0.01, ngridx)
    yi = np.linspace(min(Y["y"])-0.01, max(Y["y"])+0.01, ngridx)
    hist_range=((min(Y["x"]), max(Y["x"])), (min(Y["y"]), max(Y["y"])))
    zi,xi,yi = np.histogram2d(x=Y2["x"], y=Y2["y"], weights=Y2.color,range=hist_range,bins=(xi,yi), density=True)
    zi=zi.T
    zi=zi/np.sum(zi)

    if use_diff:
        zi2, xi2, yi2 = np.histogram2d(x=Y3["x"], y=Y3["y"], weights=Y3.color, range=hist_range, bins=(xi, yi), density=True)
        zi2 = zi2.T
        zi2 = zi2 / np.sum(zi2)
        zi=(zi-zi2)
        print("Max zi diff: {} \n Min zi diff: {}".format(np.max(zi),np.min((zi))))
        #zi[zi!=0.0]=(zi[zi!=0.0]-np.min(zi))/(np.max(zi)-np.min(zi))*2-1
    else:
        print("Max zi diff: {} \n Min zi diff: {}".format(np.max(zi), np.min((zi))))
        #zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))

    if imagemethod=="contour":
        extent = [lim_min, lim_max, lim_min, lim_max].copy()
        xi = (xi[:-1] + xi[1:]) / 2
        yi = (yi[:-1] + yi[1:]) / 2
        xi=xi.tolist()
        yi=yi.tolist()
        xi.insert(0, lim_min)
        yi.insert(0, lim_min)
        xi.append(lim_max)
        yi.append(lim_max)
        xi=np.array(xi)
        yi=np.array(yi)
        newz=np.zeros((zi.shape[0]+2,zi.shape[1]+2))
        newz[0,0]=max_zi
        newz[-1,-1]=min_zi
        newz[1:-1,1:-1]=zi.copy()

        ax.contour(xi,yi,newz, levels=int_level, linewidths=0.5)
        cntr1 = ax.contourf(xi,yi,newz, levels=int_level, cmap="RdBu_r")
    else:
        cntr1=ax.imshow(zi, extent=[xi[0], xi[-1], yi[0], yi[-1]], cmap="RdBu_r", origin="lower", interpolation=im_int_method)
    fig.colorbar(cntr1, ax=ax)


    #ax.scatter(Y2["x"],Y2["y"], alpha=0.0, s=1)
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
        ax.annotate(row.name, (row[0], row[1]), xytext=(6, 0), textcoords='offset points', alpha=0.6)


    #for idx, row in Y2.iloc[0:50,:].iterrows():
    #    ax.annotate(row.name, (row[0], row[1]), xytext=(-10, 0), textcoords='offset points', alpha=0.6)



    # force matplotlib to draw the graph
    ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    #ax.xaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    plt.show()
