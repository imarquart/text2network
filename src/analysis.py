# %% Imports


import glob
import xlsxwriter
import os, time, sys
import logging
from shutil import copyfile
from sys import exit
from NLP.src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from NLP.config.config import configuration
from NLP.src.process_sentences_network import process_sentences_network
from NLP.utils.load_bert import get_bert_and_tokenizer
from NLP.utils.hash_file import hash_file, check_step, complete_step
from NLP.src.run_bert import bert_args, run_bert
from NLP.src.dynamic_clustering import dynamic_clustering, louvain_cluster
from NLP.src.centrality_measures import dynamic_centralities
import networkx as nx
from NLP.src.reduce_network import draw_ego_network_mem, draw_ego_network

# %% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg = configuration()

# %% Main Loop:
years = range(1990, 2016)
logging.info("Setting up folder structure")
cluster_xls = ''.join([cfg.data_folder, '/cluster_xls'])
plot_folder = ''.join([cfg.data_folder, '/plots/cluster'])
egoplot_folder = ''.join([cfg.data_folder, '/plots/ego'])

if not os.path.exists(cluster_xls): os.mkdir(cluster_xls)
if not os.path.exists(plot_folder): os.mkdir(plot_folder)
if not os.path.exists(egoplot_folder): os.mkdir(egoplot_folder)

# %% Dynamic Clustering
for network_type in ["Rgraph-Sum-Rev", "Rgraph-Sum", "Cgraph-Sum", "Cgraph-Sum-Rev"]:
    for focal_token in cfg.focal_nodes:
        logging.info("%s: Clustering token %s" % (network_type, focal_token))
        central_graphs, year_info = dynamic_clustering(years, focal_token, cfg, louvain_cluster, cfg.cluster_window,
                                                       cfg.num_retain, cfg.ego_radius, network_type=network_type)
        logging.info("Saving dynamic clustering data")

        workbook = xlsxwriter.Workbook(
            ''.join([cluster_xls, '/', focal_token, '_', network_type, 'w_', str(cfg.cluster_window), '.xlsx']))

        for year in years:
            cplot_folder = ''.join([plot_folder, '/cluster_', focal_token, '_', str(year), '_', network_type, '_w_', str(cfg.cluster_window)])
            cplot_name = ''.join(["Cluster ", str(year), ' ', network_type, ' w ', str(cfg.cluster_window)])
            draw_ego_network_mem(central_graphs[year], focal_token, 15, cplot_name, cplot_folder, False)
            worksheet.write_number(0, 0, year)
            row = 1
            col = 1
            worksheet = workbook.add_worksheet()
            for cluster in year_info[year]:
                row = 1
                for token, cent in cluster.items():
                    worksheet.write_string(row, col, token)
                    worksheet.write_number(row, col + 1, cent)
                    row = row + 1

                col = col + 2

        workbook.close()

#%% Ego Plots

for year in years:
    data_folder = ''.join([cfg.data_folder, '/', str(year)])
    nw_folder=''.join([data_folder,'/networks'])
    sum_folder=''.join([nw_folder,'/sums'])
    logging.info("Drawing Ego Plots for year %i" % year)
    for focal_node in cfg.focal_nodes:
        plot_list=['Rgraph']
        for plot_type in plot_list:
            n_graph = os.path.join(sum_folder, "".join([plot_type,'-Sum.gexf']))
            r_graph = os.path.join(sum_folder, "".join([plot_type,'-Sum-Rev.gexf']))
            n_output=os.path.join(egoplot_folder, "".join([focal_node,'-',plot_type,'-',str(year),'.png']))
            r_output=os.path.join(egoplot_folder, "".join([focal_node,'-',plot_type,'-',str(year),'-Rev.png']))
            n_title=''.join([str(year),' ', plot_type,' Out Ego Network: Conditioned on ',focal_node])
            r_title=''.join([str(year),' ', plot_type,' In Ego Network: Conditioned on neighbors'])

            draw_ego_network(n_graph, focal_node, cfg.ego_limit, plot_title=n_title, save_folder=n_output,
                                 plot_screen=False)
            draw_ego_network(r_graph, focal_node, cfg.ego_limit, plot_title=r_title, save_folder=r_output,
                                 plot_screen=False)

    logging.info("Finished drawing Ego Plots")

# %% Dynamic Centralities
for network_type in ["Rgraph-Sum-Rev", "Rgraph-Sum", "Cgraph-Sum"]:
#for network_type in ["Rgraph-Sum-Rev"]:
    for focal_token in cfg.focal_nodes:
    #for focal_token in ['leader']:
        logging.info("%s: Centrality for token %s" % (network_type, focal_token))
        year_info = dynamic_centralities(years, focal_token, cfg,
                                                       cfg.num_retain, network_type=network_type)
        logging.info("Saving Centrality data")

        workbook = xlsxwriter.Workbook(
            ''.join([cluster_xls, '/centralities', focal_token, '_', network_type, '.xlsx']))

        for measure in year_info[years[0]]:
            #measure=list(measure.keys())[0]
            worksheet = workbook.add_worksheet()
            worksheet.write_string(0, 0, measure)

            # Print token names
            mdict = year_info[years[0]][measure]
            row=2
            for token,cent in mdict.items():
                worksheet.write_string(row, 0, token)
                row=row+1

            # Print token values
            col=1
            for t, year in enumerate(years):
                worksheet.write_number(1, col, year)
                row = 2
                mdict=year_info[year][measure]
                for token,cent in mdict.items():
                    worksheet.write_number(row, col, cent*100)
                    row = row + 1
                col = col + 1

        workbook.close()
