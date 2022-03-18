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

handle_leader = "drop"
#handle_leader = "switch"
#handle_leader = None

picdir = picdir + "\\hl_" + str(handle_leader) + "_"

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

# Embedding
nodes = list(X.index)
kernel_pca = KernelPCA(remove_zero_eig=False,
                       n_components=2, kernel="precomputed", random_state=100)
X.index = nodes
X.columns = nodes
Y = kernel_pca.fit_transform(X)
pos = Y.copy()

# Create Network Representation
ids = list(range(0, len(X.index)))
index_dict = dict(zip(ids, nodes))
node_dict = dict(zip(nodes, ids))
X.index = ids
X.columns = ids
G = nx.from_pandas_adjacency(X)

Y = pd.DataFrame(Y)
X.index = nodes
X.columns = nodes
Y.index = nodes
Y["color"] = [item / np.max(color) for item in color]
Y.columns = ["x", "y", "color"]

# Create figure
fig = plt.figure(figsize=(12, 10))
fig.suptitle(
    "{}-{}: Contextual clusters \n leader lvl {}, \n selected with {}".format(years[0], years[-1], 10, "weight"),
    fontsize=14
)
xlim_max = np.max(np.max(Y.loc[:, ["x"]]))
xlim_min = np.min(np.min(Y.loc[:, ["x"]]))
ylim_max = np.max(np.max(Y.loc[:, ["y"]]))
ylim_min = np.min(np.min(Y.loc[:, ["y"]]))
xlim_max = xlim_max * (1 + scale_percent)
xlim_min = xlim_min * (1 + scale_percent)
ylim_max = ylim_max * (1 + scale_percent)
ylim_min = ylim_min * (1 + scale_percent)
ax = fig.add_subplot(xlim=(xlim_min, xlim_max), ylim=(ylim_min, ylim_max))

ax.scatter(Y["x"], Y["y"], s=10, c=Y.color, alpha=0.1, cmap="Pastel1")

# Identify extremal points
extremal_points = []
extremal_points.append(Y.iloc[np.argmax(Y.x), :])
extremal_points.append(Y.iloc[np.argmax(Y.y), :])
extremal_points.append(Y.iloc[np.argmin(Y.x), :])
extremal_points.append(Y.iloc[np.argmin(Y.y), :])

set_points = list(Y.index)
starting_point = extremal_points[2].name
convex_hull = get_convex_hull(Y, set_points, starting_point)
ch_frame = pd.DataFrame(convex_hull)
ax.scatter(Y["x"], Y["y"], s=100, c=Y.color, cmap="Pastel1")

ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ftitle = picdir + "nw_graph_raw.png"
fig.savefig(ftitle)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > X.mean().mean()]
nx.draw_networkx_edges(G, pos, edgelist=elarge, ax=ax, alpha=0.1)

ftitle = picdir + "nw_graph_edges.png"
fig.savefig(ftitle)

# for idx, row in enumerate(convex_hull):
for idx, row in Y.iterrows():
    names = cldict[row.name]
    names.remove(row.name)
    names.insert(0, row.name)
    names = names[0:6]
    n = len(names)
    d = 8
    dist = n * d
    start = dist // 2
    for i, name in enumerate(names):
        if i > 0:
            alpha = 0.5
        else:
            alpha = 1
        ax.annotate(name, (row[0], row[1]), xytext=(5, start - i * d), textcoords='offset points', alpha=alpha)

ax.axis("tight")
ftitle = picdir + "nw_graph.png"
plt.savefig(ftitle)
plt.close()
##plt.show()


# %% Landscape

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

if handle_leader == "drop":
    df=df.loc[~(df["occ"] == "leader"),:].copy()
elif handle_leader == "switch":
    df.loc[df["occ"] == "leader", "occ"] = df.loc[df["occ"] == "leader", "sub"]
else:
    pass

for sel_alter in sel_alter_list:

    X2 = df[cl_name].copy()
    # X2=X2.div(X2.sum(axis=1), axis=0)
    X2["color"] = df.rweight / np.max(df.rweight)
    X2["alter"] = semantic_network.ensure_tokens(df.occ)

    X2["year"] = df.tYear
    # summedX2=X2.groupby("alter", as_index=False).sum().sort_values(by="color", ascending=False)
    # summedX2.iloc[:,1:-1]=summedX2.iloc[:,1:-1].div(summedX2.iloc[:,1:-1].sum(axis=1), axis=0)
    # X2=summedX2
    X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    X2 = X2.sort_values(by="color", ascending=False).iloc[0:nr_tokens, :]
    # Drop rows with no connections
    X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]

    if sel_alter is not None:
        XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].std().sum(axis=1)
        title = "Within Year STD {}".format(",".join(sel_alter))
        ftitle = picdir + "within-diff-" + str(sel_alter[0]) + ".png"
        XYear.to_excel(ftitle + ".xlsx")
        plt.scatter(XYear.index,XYear)
    else:
        XYear = X2.groupby("year")[cl_name].std().sum(axis=1)
        plt.scatter(XYear.index,XYear)
        title = "Within Year STD all"
        ftitle = picdir + "within-diff-all.png"
        XYear.to_excel(ftitle + ".xlsx")
    plt.title(title)
    plt.savefig(ftitle)
    plt.close()
    ##plt.show()
    #
    # if sel_alter is not None:
    #     XYear = np.abs(X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean().diff(1).sum(axis=1))
    #     title = "Summed difference from t-1 {}".format(",".join(sel_alter))
    #     ftitle = picdir + "summed-difft1-" + str(sel_alter[0]) + ".png"
    #     plt.plot(XYear)
    # else:
    #     XYear = np.abs(X2.groupby("year")[cl_name].mean().diff(1).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "Summed difference from t-1 all"
    #     ftitle = picdir + "summed-difft1-all.png"
    # plt.title(title)
    # plt.savefig(ftitle)
    # plt.close()
    # # plt.show()

    if sel_alter is not None:
        XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean()
        XYear = np.sqrt(np.square(XYear - XYear.loc[1980, :]).sum(axis=1))
        title = "L2 Difference from 1980 {}".format(",".join(sel_alter))
        ftitle = picdir + "l2-1980-" + str(sel_alter[0]) + ".png"
        plt.scatter(XYear.index,XYear)
        XYear.to_excel(ftitle + ".xlsx")
    else:
        XYear = X2.groupby("year")[cl_name].mean()
        XYear = np.sqrt(np.square(XYear - XYear.loc[1980, :]).sum(axis=1))
        plt.scatter(XYear.index,XYear)
        title = "L2 Difference from 1980 all"
        ftitle = picdir + "l2-1980.png"
        XYear.to_excel(ftitle + ".xlsx")
    plt.title(title)
    plt.savefig(ftitle)
    plt.close()
    # plt.show()
    #
    # if sel_alter is not None:
    #     XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean()
    #     XYear = np.sqrt(np.square(XYear - XYear.iloc[0:6, :].mean()).sum(axis=1))
    #     title = "L2 Difference from  1980-1985 average  {}".format(",".join(sel_alter))
    #     ftitle = picdir + "l2-1980-1985-" + str(sel_alter[0]) + ".png"
    #     plt.plot(XYear)
    # else:
    #     XYear = X2.groupby("year")[cl_name].mean()
    #     XYear = np.sqrt(np.square(XYear - XYear.iloc[0:6, :].mean()).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "L2 Difference from  1980-1985 average  all"
    #     ftitle = picdir + "l2-1980-1985.png"
    # plt.title(title)
    # plt.savefig(ftitle)
    # plt.close()
    # # plt.show()
    #
    # if sel_alter is not None:
    #     XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean()
    #     XYear = np.sqrt(np.square(XYear - XYear.iloc[0:11, :].mean()).sum(axis=1))
    #     title = "L2 Difference from  1980-1990 average  {}".format(",".join(sel_alter))
    #     ftitle = picdir + "l2-1980-1990-" + str(sel_alter[0]) + ".png"
    #     plt.plot(XYear)
    # else:
    #     XYear = X2.groupby("year")[cl_name].mean()
    #     XYear = np.sqrt(np.square(XYear - XYear.iloc[0:11, :].mean()).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "L2 Difference from  1980-1990 average  all"
    #     ftitle = picdir + "l2-1980-1900.png"
    # plt.title(title)
    # plt.savefig(ftitle)
    # plt.close()
    # # plt.show()
    #
    # if sel_alter is not None:
    #     XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean()
    #     XYear = (XYear.diff(1))
    #     XYear = np.sqrt(np.square(XYear).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "L2 Difference from t-1 {}".format(",".join(sel_alter))
    #     ftitle = picdir + "l2-t-1-" + str(sel_alter[0]) + ".png"
    # else:
    #     XYear = X2.groupby("year")[cl_name].mean()
    #     XYear = (XYear.diff(1))
    #     XYear = np.sqrt(np.square(XYear).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "L2 Difference from t-1  all"
    #     ftitle = picdir + "l2-t-1.png"
    # plt.title(title)
    # plt.savefig(ftitle)
    # plt.close()
    # # plt.show()
    #
    # if sel_alter is not None:
    #     XYear = X2[X2.alter.isin(sel_alter)].groupby("year")[cl_name].mean()
    #     XYear = (XYear.diff(1) + XYear.diff(2) + XYear.diff(3)) / 3
    #     XYear = np.sqrt(np.square(XYear).sum(axis=1))
    #     plt.plot(XYear)
    #     title = "L2 Difference from 3-MA {}".format(",".join(sel_alter))
    #     ftitle = picdir + "l2d-3ma-" + str(sel_alter[0]) + ".png"
    # else:
    #     XYear = X2.groupby("year")[cl_name].mean()
    #     XYear = (XYear.diff(1) + XYear.diff(2) + XYear.diff(3)) / 3
    #     XYear = np.sqrt(np.square(XYear).sum(axis=1))
    #     title = "L2 Difference from 3-MA all"
    #     ftitle = picdir + "l2d-3ma-all.png"
    #     plt.plot(XYear)
    # plt.title(title)
    # plt.savefig(ftitle)
    # plt.close()
    # # plt.show()

    if convex_proximity:
        subsetXcols = list(ch_frame.index)
        subset_frame = ch_frame[["x", "y"]].to_numpy()
    else:
        subsetXcols = cl_name
        subset_frame = Y[["x", "y"]].to_numpy()

    xx = X2[subsetXcols].to_numpy()
    sorted_row_idx = np.argsort(xx, axis=1)[:, 0:-emb_top_n]
    col_idx = np.arange(xx.shape[0])[:, None]
    xx[col_idx, sorted_row_idx] = 0
    xx = xx / np.sum(xx, axis=1, keepdims=True)
    Y2 = np.matmul(xx, subset_frame)
    Y2 = pd.DataFrame(Y2)

    Y.index = X.index
    Y["color"] = [item / np.max(color) for item in color]
    Y2.index = X2.index
    Y2["color"] = X2.loc[:, "color"]
    Y2["alter"] = X2["alter"]
    Y2["year"] = X2.year
    Y.columns = ["x", "y", "color"]
    Y2.columns = ["x", "y", "color", "alter", "year"]
    t1 = time()
    Y2 = Y2.dropna()
    print(np.max(Y2, axis=0))
    print(np.min(Y2, axis=0))
    Y2all = Y2.copy()

    for year in year_list + ["all"]:
        for use_diff in [True, False]:
            if sel_alter is not None:
                use_diff = True
            for time_diff in [True, False]:
                if use_diff == False:
                    time_diff = False
                if year == "all":
                    start_year = 1980
                    end_year = 2020
                else:
                    start_year = max(years[0], year - moving_average[0])
                    end_year = min(years[-1], year + moving_average[1])
                ma_years = list(np.arange(start_year, end_year + 1))
                Y2 = Y2all[Y2all.year.isin(ma_years)]

                # Create figure
                fig = plt.figure(figsize=(12, 10))
                xlim_max = np.max(np.max(Y.loc[:, ["x"]]))
                xlim_min = np.min(np.min(Y.loc[:, ["x"]]))
                ylim_max = np.max(np.max(Y.loc[:, ["y"]]))
                ylim_min = np.min(np.min(Y.loc[:, ["y"]]))
                xlim_max = xlim_max * (1 + scale_percent)
                xlim_min = xlim_min * (1 + scale_percent)
                ylim_max = ylim_max * (1 + scale_percent)
                ylim_min = ylim_min * (1 + scale_percent)
                ax = fig.add_subplot(xlim=(xlim_min, xlim_max), ylim=(ylim_min, ylim_max))

                fig.suptitle(
                    "{}-{}: Contextual clusters \n leader vs {} \n lvl {}, \n diff {}, diff to other time periods {}".format(
                        ma_years[0], ma_years[-1], sel_alter, 10, use_diff, time_diff), fontsize=12
                )

                if sel_alter is not None:
                    print("Orig Nr of Observations in time-window: {}".format(len(Y2)))
                    if use_diff:
                        Y3 = Y2.copy()
                        if sel_alter2 is not None:
                            Y3 = Y3[Y3.alter.isin(sel_alter2)]
                            print("Selecting against alternate group with nr Observations: {}".format(len(Y3)))
                        elif time_diff:
                            Y3 = Y2all[~Y2all.year.isin(ma_years)]
                            print("Orig Nr of Observations out of time-window: {}".format(len(Y3)))
                            Y3 = Y3[Y3.alter.isin(sel_alter)]
                            print("Selecting Observations  out of time-window: {}".format(len(Y3)))
                    else:
                        Y3 = None
                    Y2 = Y2[Y2.alter.isin(sel_alter)]
                    print("Selected of Observations: {}".format(len(Y2)))
                else:
                    if use_diff:
                        Y3 = Y2all.copy()

                xi = np.linspace(xlim_min, xlim_max, ngridx)
                yi = np.linspace(ylim_min, ylim_max, ngridx)

                hist_range = ((xlim_min, xlim_max), (ylim_min, ylim_max))
                zi, xi, yi = np.histogram2d(x=Y2["x"], y=Y2["y"], weights=Y2.color, range=hist_range, bins=(xi, yi),
                                            density=True)
                zi = zi.T
                zi = zi / np.sum(zi)

                if use_diff:
                    zi2, xi2, yi2 = np.histogram2d(x=Y3["x"], y=Y3["y"], weights=Y3.color, range=hist_range,
                                                   bins=(xi, yi), density=True)
                    zi2 = zi2.T
                    zi2 = zi2 / np.sum(zi2)
                    zi = (zi - zi2)
                    print("Max zi diff: {} \n Min zi diff: {}".format(np.max(zi), np.min((zi))))
                    # zi[zi!=0.0]=(zi[zi!=0.0]-np.min(zi))/(np.max(zi)-np.min(zi))*2-1
                else:
                    print("Max zi diff: {} \n Min zi diff: {}".format(np.max(zi), np.min((zi))))
                    # zi = (zi - np.min(zi)) / (np.max(zi) - np.min(zi))

                max_zi = np.max([np.max(zi), np.abs(np.min(zi))])
                if use_diff == True:
                    min_zi = -max_zi
                else:
                    min_zi = np.min(zi)

                if imagemethod == "contour":
                    # extent = [lim_min, lim_max, lim_min, lim_max].copy()
                    xi = (xi[:-1] + xi[1:]) / 2
                    yi = (yi[:-1] + yi[1:]) / 2
                    xi = xi.tolist()
                    yi = yi.tolist()
                    xi.insert(0, xlim_min)
                    yi.insert(0, ylim_min)
                    xi.append(xlim_max)
                    yi.append(ylim_max)
                    xi = np.array(xi)
                    yi = np.array(yi)
                    newz = np.zeros((zi.shape[0] + 2, zi.shape[1] + 2))
                    newz[0, 0] = max_zi
                    newz[-1, -1] = min_zi
                    newz[1:-1, 1:-1] = zi.copy()

                    ax.contour(xi, yi, newz, levels=int_level, linewidths=0.5)
                    cntr1 = ax.contourf(xi, yi, newz, levels=int_level, cmap="RdBu_r")
                else:
                    cntr1 = ax.imshow(zi, extent=[xi[0], xi[-1], yi[0], yi[-1]], cmap="RdBu_r", origin="lower",
                                      interpolation=im_int_method)
                fig.colorbar(cntr1, ax=ax)

                # ax.scatter(Y2["x"],Y2["y"], alpha=0.0, s=1)
                ax.scatter(Y["x"], Y["y"], s=100, c=Y.color, cmap="Pastel1")
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.axis("tight")
                if sel_alter is None:
                    ftitle = picdir + "lsall_d{}_t{}".format(use_diff, time_diff) + str(ma_years[0])[-2:] + str(
                        ma_years[-1])[-2:] + ".png"
                else:
                    ftitle = picdir + "ls{}_d{}_t{}".format(sel_alter[0], use_diff, time_diff) + str(ma_years[0])[
                                                                                                   -2:] + str(
                        ma_years[-1])[-2:] + ".png"

                fig.savefig(ftitle)

                for idx, row in Y.iterrows():
                    names = cldict[row.name]
                    names.remove(row.name)
                    names.insert(0, row.name)
                    names = names  # [0:4]
                    n = len(names)
                    d = 8
                    dist = n * d
                    start = dist // 2
                    for i, name in enumerate(names):
                        if i > 0:
                            alpha = 0.5
                        else:
                            alpha = 1
                        ax.annotate(name, (row[0], row[1]), xytext=(5, start - i * d), textcoords='offset points',
                                    alpha=alpha)

                if sel_alter is None:
                    ftitle = picdir + "alsall_d{}_t{}".format(use_diff, time_diff) + str(ma_years[0])[-2:] + str(
                        ma_years[-1])[-2:] + ".png"
                else:
                    ftitle = picdir + "als{}_d{}_t{}".format(sel_alter[0], use_diff, time_diff) + str(ma_years[0])[
                                                                                                    -2:] + str(
                        ma_years[-1])[-2:] + ".png"

                fig.savefig(ftitle)
                plt.close(fig)
                # plt.show()
