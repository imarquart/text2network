import copy
import json
import logging
import os
import time

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config.dataclasses import TrainingConfig
from text2network.datasets.text_datasets import PreTrainDataSet
from text2network.functions.train_mlm import train_mlm_simple
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.hash_file import check_step, complete_step, hash_string
from text2network.utils.logging_helpers import log

logger = logging.getLogger("t2n")


class model_trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logging_level = config.logging_level
        self.other_loggers = config.other_loggers

    @property
    def tokenizer_base_folder(self):
        assert self.config.output_folder, "Please initialize config.output_folder!"
        return check_create_folder(os.path.join(self.config.output_folder, "base_model/tokenizer/"))

    @property
    def model_base_folder(self):
        assert self.config.output_folder, "Please initialize config.output_folder!"
        return check_create_folder(os.path.join(self.config.output_folder, "base_model/model/"))

    @log()
    def prep_tokenizer(self):
        """Get rid of WordPieces in a BERT Tokenizer.
        Save pretrained tokenizer in old format (Python implementation)
        Modify vocab file, then load tokenizer via FastTokenizer package.
        """
        model = self.config.llm.model_name_or_path
        # Can't use fast tokenizer because we need to write the vocab file
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        logger.info("Tokenizer vocabulary {} items.".format(len(tokenizer)))

        tokenizer_folder = self.tokenizer_base_folder + "/tmp"
        tokenizer.save_pretrained(tokenizer_folder)
        new_vocab = {}
        for i, row in enumerate(open(tokenizer_folder + "/vocab.txt", "r")):
            new_vocab[i] = row.strip().replace("##", "@@@@")
        with open(tokenizer_folder + "/vocab.txt", "w") as f:
            for i in range(len(new_vocab)):
                f.write(new_vocab[i] + "\n")
        new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder, use_fast=True)
        logger.debug(f"De-Wordpieced Tokenizer has {len(new_tokenizer)} tokens.")

        return new_tokenizer

    @log()
    def prep_model(self, tokenizer):
        """Load a BERT model, resize it with new tokens, then save it to the base dictionary.

        Args:
            tokenizer: Tokenizer with added tokens
        """
        model = AutoModelForMaskedLM.from_pretrained(self.config.llm.model_name_or_path)
        logger.info("Resizing embeddings of BERT model")
        model.resize_token_embeddings(len(tokenizer))
        model.save_pretrained(self.model_base_folder)

        logger.info("Test-loading model")
        try:
            test_model = AutoModelForMaskedLM.from_pretrained(self.model_base_folder)
        except Exception as e:
            logger.error(f"Failed to load model after resizing: {e}")
            raise e

        return model

    @log()
    def get_consistent_tokenizer(self):
        """Initializes tokenizer without WordPiece, finds missing tokens in dataset
        and adds them, then saves the tokenizer into one place.

        Returns:
            tokenizer
        """
        tokenizer = self.prep_tokenizer()
        pts = PreTrainDataSet(self.config.data)
        logger.debug(f"Found llm's to train in dataset {pts.llm}")
        missing_tokens = pts.get_missing_tokens(llm_list=pts.llms, tokenizer=tokenizer)
        missing_tokens = missing_tokens.index.to_list()
        logger.info(f"Found {len(missing_tokens)} missing tokens to be added to the tokenizer")
        tokenizer.add_tokens(missing_tokens)
        logger.info(
            "After adding missing terms: Tokenizer vocabulary {} items.".format(len(tokenizer))
        )
        logger.info(f"Saving tokenizer to {self.tokenizer_base_folder}.")
        tokenizer.save_pretrained(self.tokenizer_base_folder)

        return tokenizer

    @log()
    def prep_mlm_and_tokenizer(self):
        """
        Preps both model and tokenizer and writes into
        """
        hash = hash_string(self.tokenizer_base_folder, hash_factory="md5")
        if check_step(self.tokenizer_base_folder, hash):
            logger.info(f"Pre-populated tokenizer found in {self.tokenizer_base_folder}. Using!")
            new_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_base_folder, use_fast=True)
            logger.info("Loaded Tokenizer vocabulary {} items.".format(len(new_tokenizer)))
        else:
            logger.info("Pre-Loading Data and Populating tokenizers")
            new_tokenizer = self.get_consistent_tokenizer()
            assert new_tokenizer is not None, "Tokenizer prep failed"
            complete_step(self.tokenizer_base_folder, hash)

        hash = hash_string(self.model_base_folder, hash_factory="md5")
        if check_step(self.model_base_folder, hash):
            logger.info(f"Pre-prepped model found in {self.model_base_folder}. Using!")
        else:
            new_model = self.prep_model(new_tokenizer)
            assert new_model is not None, "Model prep failed"
            complete_step(self.model_base_folder, hash)

    def train_one_model(self, llm: str):
        """Preps a single model to train for a given llm setting present in the database.

        Args:
            llm: A string that gives a llm to train.
            The llm is a folder that the PretrainedDataset can read, i.e. it's
            present under PretrainedDataset.llms

        Returns:
            Results of the run
        """
        config = copy.deepcopy(self.config)
        # Create config file for this run.
        config.llm.model_name_or_path = self.model_base_folder
        config.llm.tokenizer_name_or_path = self.tokenizer_base_folder
        output_folder = self.compute_outputfolder_for_llm(config, llm=llm)
        config.llm.model_output_folder = output_folder
        config.llm.llm = llm

        logger.info(
            f"Starting training for model \n"
            f"Base model: {config.llm.model_name_or_path} \n"
            f"Base tokenizer: {config.llm.tokenizer_name_or_path} \n"
            f"Trained model output to: {config.llm.model_output_folder} \n"
        )

        torch.cuda.empty_cache()
        results = train_mlm_simple(config)
        return results

    def train_models(self):
        # Create tokenizer and model without Wordpieces and added missing tokens
        self.prep_mlm_and_tokenizer()
        # Initialize dataset and grab llms
        pts = PreTrainDataSet(self.config.data)
        llm_list = pts.llms

        # Train Models
        logger.info(
            "With the current hierarchy, there are %i BERT models to train",
            (len(llm_list)),
        )
        for idx, llm in enumerate(llm_list):
            bert_folder = self.compute_outputfolder_for_llm(self.config, llm=llm)
            logger.info("----------------------------------------------------------------------")

            # Set up Hash
            hash = hash_string(bert_folder, hash_factory="md5")

            if check_step(bert_folder, hash):
                logger.info("Found trained model for %s. Skipping", bert_folder)
            else:
                start_time = time.time()
                results = self.train_one_model(llm=llm)
                logger.info("Model training finished in %s seconds", (time.time() - start_time))
                logger.info("%s: Model results %s", llm, results)
                complete_step(bert_folder, hash)
                logger.info(
                    "----------------------------------------------------------------------"
                )

    @staticmethod
    def compute_outputfolder_for_llm(config, llm):
        return check_create_folder(os.path.join(config.output_folder, f"llms/{llm}"))
