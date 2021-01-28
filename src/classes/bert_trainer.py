import json

from src.functions.file_helpers import check_create_folder
from src.utils.hash_file import hash_string, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer
from src.utils.bert_args import bert_args
from src.datasets.text_dataset import bert_dataset
from src.functions.run_bert import run_bert
from src.utils.get_uniques import get_uniques
import tables
import torch
import logging
import time


class bert_trainer():
    def __init__(self, config=None,db_folder=None, pretrained_folder=None, trained_folder=None, bert_config=None, split_hierarchy=None,
                 logging_level=None):


        # Fill parameters from configuration file
        if logging_level is not None:
            self.logging_level=logging_level
        else:
            if config is not None:
                self.logging_level=config['General'].getint('logging_level')
            else:
                msg="Please provide valid logging level."
                logging.error(msg)
                raise AttributeError(msg)
        # Set logging level
        logging.disable(self.logging_level)

        if db_folder is not None:
            self.db_folder=db_folder
        else:
            if config is not None:
                self.db_folder=config['Paths']['database']
            else:
                msg="Please provide valid database path."
                logging.error(msg)
                raise AttributeError(msg)

        if pretrained_folder is not None:
            self.pretrained_folder=pretrained_folder
        else:
            if config is not None:
                self.pretrained_folder=config['Paths']['pretrained_bert']
            else:
                msg="Please provide valid pretrained_folder."
                logging.error(msg)
                raise AttributeError(msg)


        if trained_folder is not None:
            self.trained_folder=trained_folder
        else:
            if config is not None:
                self.trained_folder=config['Paths']['trained_berts']
            else:
                msg="Please provide valid trained_berts folder."
                logging.error(msg)
                raise AttributeError(msg)

        if bert_config is not None:
            self.bert_config=bert_config
        else:
            if config is not None:
                self.bert_config=config['BertTraining']
            else:
                msg="Please provide valid bert_config."
                logging.error(msg)
                raise AttributeError(msg)

        if split_hierarchy is not None:
            self.split_hierarchy=split_hierarchy
            self.uniques = self.get_uniques(self.split_hierarchy)
        else:
            if config is not None:
                self.split_hierarchy=json.loads(config.get('General', 'split_hierarchy'))
                self.uniques = self.get_uniques(self.split_hierarchy)
            else:
                self.uniques = None
                self.split_hierarchy = None

        # Check and create folders
        check_create_folder(self.trained_folder)

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



        # Load necessary tokens
        missing_tokens = []
        logging.info("Pre-Loading Data and Populating tokenizers")
        import nltk
        nltk.download('stopwords')
        for idx, query_filename in enumerate(self.uniques["query_filename"]):
            query = query_filename[0]
            fname = query_filename[1]
            bert_folder = ''.join([self.trained_folder, '/', fname])

            # Prepare BERT and vocabulary
            tokenizer, bert = get_bert_and_tokenizer(self.pretrained_folder, True)
            dataset = bert_dataset(tokenizer, self.db_folder, query,
                                   block_size=self.bert_config.getint('max_seq_length'),check_vocab=True,freq_cutoff=self.bert_config.getint('new_word_cutoff'),
                                   logging_level=logging.DEBUG)
            missing_tokens.extend(dataset.missing_tokens)

        missing_tokens=list(set(missing_tokens))
        # Setting up tokenizer
        # We do this here to keep the IDs and Tokens consistent, although the network is able to translate
        # if necessary
        tokenizer, _ = get_bert_and_tokenizer(self.pretrained_folder, True)
        # Add missing tokens
        logging.info("Tokenizer vocabulary {} items.".format(len(tokenizer)))
        logging.disable(logging.ERROR)
        tokenizer.add_tokens(missing_tokens)
        logging.disable(self.logging_level)
        logging.info("After adding missing terms: Tokenizer vocabulary {} items.".format(len(tokenizer)))

        # Train BERTS
        logging.info("With the current hierarchy, there are %i BERT models to train" % (len(self.uniques["query"])))
        for idx, query_filename in enumerate(self.uniques["query_filename"]):
            logging.disable(self.logging_level)
            logging.info("----------------------------------------------------------------------")
            # Set up BERT folder
            query=query_filename[0]
            fname = query_filename[1]
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
                                 logging_level=logging.INFO)

                # Prepare BERT and vocabulary
                logging.info("Before resizing, Tokenizer vocabulary {} items.".format(len(tokenizer)))
                # Make sure old model is deleted!
                del bert

                _, bert = get_bert_and_tokenizer(self.pretrained_folder, True)
                # Make a copy just to be safe
                new_tokenizer=tokenizer

                bert.resize_embedding_and_fc(len(new_tokenizer))
                logging.info("After resizing, Tokenizer vocabulary {} items.".format(len(new_tokenizer)))

                logging.info("Training BERT on %s" % (query))
                start_time = time.time()
                torch.cuda.empty_cache()
                #logging.disable(logging.ERROR)
                results = run_bert(args, tokenizer=new_tokenizer, model=bert)
                logging.disable(self.logging_level)
                logging.info("BERT training finished in %s seconds" % (time.time() - start_time))
                logging.info("%s: BERT results %s" % (fname, results))
                complete_step(bert_folder, hash)
                logging.info("----------------------------------------------------------------------")
