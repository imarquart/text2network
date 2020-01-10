# TODO: Comment
import logging
class configuration:
    """
    Simple class to store our options
    """
    #%% Analysis options:
    years=range(1991,2019)
    prune_min=0.025
    cluster_windows=[2,3]
    ego_radius=1
    num_retain=150
    num_retain_cluster=150
    cluster_levels=3
    save_cluster_to_xlsx = True
    #%% Plot options:
    focal_nodes=["leader","leadership",'manager','management']
    ego_limit=20

    #%% Folders:
    model_dir = "D:/NLP/BERT-NLP/NLP/models"
    data_folder = "E:/NLP"
    text_folder = "D:/NLP/BERT-NLP/NLP/data"
    nw_folder="/networksNoCut"
    sums_folder=''.join([nw_folder,'/sums'])
    ma_folder=''.join([nw_folder,'/MAsums'])
    np_folder = ''.join([nw_folder, '/np_plural_sums'])
    entropy_folder=''.join([nw_folder,'/entropy'])
    cluster_xls = ''.join([data_folder,'/cluster_xls'])
    plot_folder = ''.join([data_folder,'/plots/'])
    clusterplot_folder = ''.join([data_folder,'/plots/'])
    egoplot_folder = ''.join([data_folder,'/plots/ego'])
    #%% Logging
    subprocess_level=logging.INFO


    #%% Text Preprocessing options
    max_seq_length = 30
    char_mult=10
    max_seq=0

    #%% BERT Training options
    gpu_batch=100
    epochs=100
    loss_limit=0.5
    warmup_steps = 0
    mlm_probability = 0.15
    do_train=True

    #%% BERT Inference Options
    batch_size=30


    #%% Network creation options
    cutoff_percent = 95
    max_degree=20
    plural_method="sum"

    #%% Moving Average options
    ma_order=3
    average_links=True
