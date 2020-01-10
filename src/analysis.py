# %% Imports

import pandas as pd
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
from NLP.src.dynamic_clustering import dynamic_clustering, louvain_cluster, overall_clustering
from NLP.src.centrality_measures import dynamic_centralities
import networkx as nx
from NLP.src.draw_networks import draw_ego_network_mem, draw_ego_network

# %% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg = configuration()

# %% Main Loop:
years = cfg.years
logging.info("Setting up folder structure")
cluster_xls = ''.join([cfg.data_folder, '/cluster_xls'])
plot_folder = ''.join([cfg.data_folder, '/plots/cluster'])
egoplot_folder = ''.join([cfg.data_folder, '/plots/ego'])
centralities_folder = ''.join([cluster_xls, '/centralities'])

if not os.path.exists(cluster_xls): os.mkdir(cluster_xls)
if not os.path.exists(plot_folder): os.mkdir(plot_folder)
if not os.path.exists(egoplot_folder): os.mkdir(egoplot_folder)
if not os.path.exists(centralities_folder): os.mkdir(centralities_folder)

# %% Ego Plots

for year in years:
    data_folder = ''.join([cfg.data_folder, '/', str(year)])
    nw_folder = ''.join([data_folder, '/networks'])
    sum_folder = ''.join([nw_folder, '/sums'])
    logging.info("Drawing Ego Plots for year %i" % year)
    for focal_node in cfg.focal_nodes:
        plot_list = ['Rgraph']
        for plot_type in plot_list:
            n_graph = os.path.join(sum_folder, "".join([plot_type, '-Sum.gexf']))
            r_graph = os.path.join(sum_folder, "".join([plot_type, '-Sum-Rev.gexf']))
            n_output = os.path.join(egoplot_folder, "".join([focal_node, '-', plot_type, '-', str(year), '.png']))
            r_output = os.path.join(egoplot_folder, "".join([focal_node, '-', plot_type, '-', str(year), '-Rev.png']))
            n_title = ''.join([str(year), ' ', plot_type, ' Out Ego Network: Conditioned on ', focal_node])
            r_title = ''.join([str(year), ' ', plot_type, ' In Ego Network: Conditioned on neighbors'])
            if not os.path.exists(n_output):
                draw_ego_network(n_graph, focal_node, cfg.ego_limit, plot_title=n_title, save_folder=n_output,
                             plot_screen=False)
            if not os.path.exists(r_output):
                draw_ego_network(r_graph, focal_node, cfg.ego_limit, plot_title=r_title, save_folder=r_output,
                             plot_screen=False)

    logging.info("Finished drawing Ego Plots")

# %% Dynamic Centralities

for network_type in ["Rgraph-Sum-Rev"]:
    # for network_type in ["Rgraph-Sum-Rev"]:
    for focal_token in cfg.focal_nodes:
        # for focal_token in ['leader']:
        logging.info("%s: Centrality for token %s" % (network_type, focal_token))
        network_file = ''.join([network_type,'_order',str(cfg.ma_order)])
        filename=''.join([centralities_folder,'/', focal_token, '_', network_type, '_order',str(cfg.ma_order),'.xlsx'])
        if not os.path.exists(filename):
            year_info = dynamic_centralities(years, focal_token, cfg,
                                             cfg.num_retain, network_type=network_file, sums_folder=cfg.ma_folder)
            logging.info("Saving Centrality data in  %s" % filename)

            workbook = xlsxwriter.Workbook(filename)

            for measure in year_info[years[0]]:
                # measure=list(measure.keys())[0]
                worksheet = workbook.add_worksheet()
                worksheet.write_string(0, 0, measure)
                # Print token names
                mdict = year_info[years[0]][measure]
                row = 2
                for token, cent in mdict.items():
                    worksheet.write_string(row, 0, token)
                    row = row + 1
                # Print token values
                col = 1
                for t, year in enumerate(years):
                    worksheet.write_number(1, col, year)
                    row = 2
                    mdict = year_info[year][measure]
                    for token, cent in mdict.items():
                        worksheet.write_number(row, col, cent * 100)
                        row = row + 1
                    col = col + 1
            workbook.close()
        else:
            logging.info("%s: Centrality for token %s already exists!" % (network_type, focal_token))
# %% Dynamic Clustering
#
#for network_type in ["Rgraph-Sum-Rev","Cgraph-Sum-Rev"]:
for network_type in ["Rgraph-Sum-Rev"]:
    for cluster_window in cfg.cluster_windows:
        for focal_token in cfg.focal_nodes:
            logging.info("%s: Clustering token %s, window %i" % (network_type, focal_token, cluster_window))
            network_file = ''.join([network_type, '_order', str(cfg.ma_order)])
            year_info = dynamic_clustering(years, focal_token, cfg, louvain_cluster, cfg.ma_folder, cluster_window,
                                           cfg.num_retain_cluster, cfg.ego_radius, network_type=network_file,
                                           cluster_levels=cfg.cluster_levels)
            filename = ''.join([cluster_xls, '/', focal_token, '_', network_type, 'w_', str(cluster_window),'_order', str(cfg.ma_order),  '.csv'])
            logging.info("Saving dynamic clustering data in %s" % filename)

            #PANDAS
            c_token = []
            c_tcluster = []
            c_cluster = []
            c_level = []
            c_proximity = []
            c_spath = []
            c_cent = []
            c_ftoken = []
            c_coTo = []
            c_coFrom = []
            c_indeg=[]
            c_outdeg=[]
            c_year = []
            c_egocent=[]
            for year in years:
                info = year_info[year]
                nr_clusters = len(info)
                for level in range(0, cfg.cluster_levels):
                    for i in range(0, nr_clusters):
                        nodes = info[i]['nest'][level]
                        node_info = info[i]['info']
                        for j, subclusters in enumerate(nodes):
                            for token in subclusters:
                                c_token.append(token)
                                c_tcluster.append(i)
                                c_cluster.append(j)
                                c_level.append(level)
                                c_proximity.append(node_info[token][0])
                                c_spath.append(node_info[token][1])
                                c_cent.append(node_info[token][2])
                                c_ftoken.append(focal_token)
                                c_year.append(year)
                                c_coTo.append(node_info[token][3])
                                c_coFrom.append(node_info[token][4])
                                c_egocent.append(node_info[token][5])
                                c_indeg.append(node_info[token][6])
                                c_outdeg.append(node_info[token][7])


            data = {'Year':c_year,'Token': c_token,
                    'TopCluster': c_tcluster,
                    'Cluster': c_cluster, 'Level': c_level, 'Proximity': c_proximity, 'ShortestPath': c_spath,
                    'Indegree': c_indeg, 'Outdegree': c_outdeg,
                    'EgoCentrality': c_egocent,'ClusterCentrality': c_cent,'ConstraintToFocal':c_coTo,'ConstraintFromFocal':c_coFrom, 'FocalToken': c_ftoken}

            dframe = pd.DataFrame(data)
            dframe.to_csv(filename)
            # XLSX
            if cfg.save_cluster_to_xlsx == True:
                for year in years:
                    workbook = xlsxwriter.Workbook(
                        ''.join([cluster_xls, '/', focal_token, '_', network_type, 'w_', str(cluster_window), '.xlsx']))

                    # node info has (proximity,distance,centrality)
                    for year in years:
                        info = year_info[year]
                        worksheet = workbook.add_worksheet()
                        worksheet.name = str(year)
                        worksheet.write_number(0, 0, year)
                        row = 1
                        col = 1
                        max_row = row
                        nr_clusters = len(info)
                        for level in range(0, cfg.cluster_levels):
                            worksheet.write_string(max_row + 1, 0, ''.join(["Level ", str(level)]))
                            col = 1
                            if level == 0:  # Top level cluster
                                for i in range(0, nr_clusters):
                                    # Each cluster starts at row=1
                                    row = max_row + 1
                                    worksheet.write_string(row, col, ''.join(["Cluster ", str(i)]))
                                    row = row + 1
                                    nodes = info[i]['nest'][0]
                                    node_info = info[i]['info']
                                    for token in nodes[0]:
                                        worksheet.write_string(row, col, token)
                                        worksheet.write_number(row, col + 1, node_info[token][0])
                                        row = row + 1
                                    col = col + 4
                                max_row = max(max_row, row)
                            else:
                                orig_row = max_row + 1

                                for i in range(0, nr_clusters):
                                    # Each cluster starts at row=orig_row
                                    # Stagger

                                    sub_row = orig_row + i
                                    # Add space

                                    nodes = info[i]['nest'][level]
                                    node_info = info[i]['info']
                                    for subclusters in nodes:
                                        row = sub_row
                                        worksheet.write_string(row, col, ''.join(["Cluster ", str(i)]))
                                        row = row + 1
                                        for token in subclusters:
                                            worksheet.write_string(row, col, token)
                                            worksheet.write_number(row, col + 1, node_info[token][0])
                                            row = row + 1
                                        max_row = max(row, max_row)
                                        col = col + 2
                                    # Push next cluster out
                                    col = col + 1
                    workbook.close()


# %% Overall Clustering
#
for network_type in ["Rgraph-Sum-Rev"]:
        for focal_token in cfg.focal_nodes:
            logging.info("%s: Overall Clustering token %s" % (network_type, focal_token))
            network_file = ''.join([network_type, '_order', str(0)])
            year_info = overall_clustering(years, focal_token, cfg, louvain_cluster, cfg.sums_folder,
                                           cfg.num_retain_cluster, cfg.ego_radius, network_type=network_type,
                                           cluster_levels=cfg.cluster_levels)
            filename = ''.join([cluster_xls, '/', focal_token, '_overall_', network_type,'.csv'])
            logging.info("Saving dynamic clustering data in %s" % filename)

            # PANDAS
            c_token = []
            c_tcluster = []
            c_cluster = []
            c_level = []
            c_proximity = []
            c_spath = []
            c_cent = []
            c_ftoken = []
            c_coTo = []
            c_coFrom = []
            c_indeg = []
            c_outdeg = []
            c_year = []
            c_egocent = []
            info = year_info
            nr_clusters = len(info)
            for level in range(0, cfg.cluster_levels):
                for i in range(0, nr_clusters):
                    nodes = info[i]['nest'][level]
                    node_info = info[i]['info']
                    for j, subclusters in enumerate(nodes):
                        for token in subclusters:
                            c_token.append(token)
                            c_tcluster.append(i)
                            c_cluster.append(j)
                            c_level.append(level)
                            c_proximity.append(node_info[token][0])
                            c_spath.append(node_info[token][1])
                            c_cent.append(node_info[token][2])
                            c_ftoken.append(focal_token)
                            c_year.append(0)
                            c_coTo.append(node_info[token][3])
                            c_coFrom.append(node_info[token][4])
                            c_egocent.append(node_info[token][5])
                            c_indeg.append(node_info[token][6])
                            c_outdeg.append(node_info[token][7])

        data = {'Year': 0, 'Token': c_token,
                'TopCluster': c_tcluster,
                'Cluster': c_cluster, 'Level': c_level, 'Proximity': c_proximity, 'ShortestPath': c_spath,
                'Indegree': c_indeg, 'Outdegree': c_outdeg,
                'EgoCentrality': c_egocent, 'ClusterCentrality': c_cent, 'ConstraintToFocal': c_coTo,
                'ConstraintFromFocal': c_coFrom, 'FocalToken': c_ftoken}

        dframe = pd.DataFrame(data)
        dframe.to_csv(filename)
        # XLSX
        if cfg.save_cluster_to_xlsx == True:
            for year in years:
                workbook = xlsxwriter.Workbook(
                    ''.join([cluster_xls, '/', focal_token, '_', network_type, 'w_', str(cluster_window), '.xlsx']))

                # node info has (proximity,distance,centrality)
                for year in years:
                    info = year_info[year]
                    worksheet = workbook.add_worksheet()
                    worksheet.name = str(year)
                    worksheet.write_number(0, 0, year)
                    row = 1
                    col = 1
                    max_row = row
                    nr_clusters = len(info)
                    for level in range(0, cfg.cluster_levels):
                        worksheet.write_string(max_row + 1, 0, ''.join(["Level ", str(level)]))
                        col = 1
                        if level == 0:  # Top level cluster
                            for i in range(0, nr_clusters):
                                # Each cluster starts at row=1
                                row = max_row + 1
                                worksheet.write_string(row, col, ''.join(["Cluster ", str(i)]))
                                row = row + 1
                                nodes = info[i]['nest'][0]
                                node_info = info[i]['info']
                                for token in nodes[0]:
                                    worksheet.write_string(row, col, token)
                                    worksheet.write_number(row, col + 1, node_info[token][0])
                                    row = row + 1
                                col = col + 4
                            max_row = max(max_row, row)
                        else:
                            orig_row = max_row + 1

                            for i in range(0, nr_clusters):
                                # Each cluster starts at row=orig_row
                                # Stagger

                                sub_row = orig_row + i
                                # Add space

                                nodes = info[i]['nest'][level]
                                node_info = info[i]['info']
                                for subclusters in nodes:
                                    row = sub_row
                                    worksheet.write_string(row, col, ''.join(["Cluster ", str(i)]))
                                    row = row + 1
                                    for token in subclusters:
                                        worksheet.write_string(row, col, token)
                                        worksheet.write_number(row, col + 1, node_info[token][0])
                                        row = row + 1
                                    max_row = max(row, max_row)
                                    col = col + 2
                                # Push next cluster out
                                col = col + 1
                workbook.close()
