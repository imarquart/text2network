# TODO: Comment
import logging
class configuration:
    """
    Simple class to store our options
    """

    #%% Folders:
    model_dir = "D:/NLP/BERT-NLP/NLP/models"
    data_folder = "E:/NLP"
    text_folder = "D:/NLP/BERT-NLP/NLP/data"

    #%% Logging
    subprocess_level=logging.INFO


    #%% Text Preprocessing options
    max_seq_length = 30
    char_mult=10
    max_seq=0

    #%% BERT Training options
    gpu_batch=100
    epochs=400
    warmup_steps = 0
    mlm_probability = 0.15
    do_train=True

    #%% BERT Inference Options
    batch_size=36


    #%% Network creation options
    cutoff_percent = 80
