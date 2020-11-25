from src.utils.hash_file import hash_string, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer
from src.utils.bert_args import bert_args
from src.datasets.text_dataset import bert_dataset
from src.classes.run_bert import run_bert
from src.utils.get_uniques import get_uniques
import tables
import torch
import logging
import time


class bert_trainer():
    def __init__(self, db_folder, pretrained_folder, trained_folder, bert_config, split_hierarchy=None,
                 logging_level=logging.NOTSET):

        # Set logging level
        self.logging_level = logging_level
        logging.disable(logging_level)

        self.db_folder = db_folder
        self.pretrained_folder = pretrained_folder
        self.trained_folder = trained_folder
        self.bert_config = bert_config

        # Load Tokenizer
        # logging.disable(logging.ERROR)
        # logging.disable(logging_level)

        if split_hierarchy is not None:
            self.uniques = self.get_uniques(split_hierarchy)
            self.split_hierarchy = split_hierarchy
        else:
            self.uniques = None
            self.split_hierarchy = None

    def get_uniques(self, split_hierarchy):
        """
        Queries database to get unique values according to hierarchy provided.
        Determines how many models we would like to train.
        :param split_hierarchy: List of table parameters
        :return: dict including unique values, query strings and bert folder names
        """
        # Create hierarchy splits
        return get_uniques(split_hierarchy, self.db_folder)

    def train_berts(self, split_hierarchy=None):

        # If uniques are not defined, create them according to provided split_hierarchy
        if self.uniques == None or split_hierarchy is not None:
            assert split_hierarchy is not None
            self.uniques = self.get_uniques(split_hierarchy)

        logging.info("With the current hierarchy, there are %i BERT models to train" % (len(self.uniques["query"])))

        for idx, query in enumerate(self.uniques["query"]):
            logging.disable(self.logging_level)
            logging.info("----------------------------------------------------------------------")
            # Set up BERT folder
            fname = self.uniques["file"][idx]
            bert_folder = ''.join([self.trained_folder, '/', fname])

            # Set up Hash
            hash = hash_string(bert_folder, hash_factory="md5")

            if (check_step(bert_folder, hash)):
                logging.info("Found trained BERT for %s. Skipping" % bert_folder)
            else:

                args = bert_args(self.db_folder, query, bert_folder, self.pretrained_folder,
                                 mlm_probability=self.bert_config.getfloat('mlm_probability'),
                                 block_size=self.bert_config.getint('max_seq_length'),
                                 loss_limit=self.bert_config.getfloat('loss_limit'),
                                 gpu_batch=self.bert_config.getint('gpu_batch'),
                                 epochs=self.bert_config.getint('epochs'),
                                 warmup_steps=self.bert_config.getint('warmup_steps'),
                                 save_steps=self.bert_config.getint('save_steps'),
                                 eval_steps=self.bert_config.getint('eval_steps'),
                                 eval_loss_limit=self.bert_config.getfloat('eval_loss_limit'),
                                 logging_level=logging.ERROR)

                logging.info("Training BERT on %s" % (query))
                start_time = time.time()
                torch.cuda.empty_cache()
                logging.disable(logging.ERROR)
                results = run_bert(args)
                logging.disable(self.logging_level)
                logging.info("BERT training finished in %s seconds" % (time.time() - start_time))
                logging.info("%s: BERT results %s" % (fname, results))
                complete_step(bert_folder, hash)
                logging.info("----------------------------------------------------------------------")
