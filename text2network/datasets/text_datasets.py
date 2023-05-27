import logging
import numbers
import os
import pickle
from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from nltk.corpus import stopwords
from nltk.tag import map_tag, pos_tag
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config.dataclasses import TrainDataConfig
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import log, setup_logger

# nltk.download("stopwords")


# Setup logging
logger = logging.getLogger("t2n")


class PreTrainDataSet(object):
    """
    This dataset loads the data before training and determines two things:
    - The vocabulary to be added to the default LLM vocabulary
    - The split hierarchy and number of LLMs that need to be trained
    """

    @log(logging_level=20, other_loggers=30)
    def __init__(
        self, config: TrainDataConfig, logging_level=20, other_loggers=30, fixed_llm: str = None
    ):
        self.config = config
        self.data_path = config.data_path
        self.logging_level = logging_level
        self.other_loggers = other_loggers
        self.cache_dir = config.cache_dir
        self.val_items = config.val_items
        self.new_word_cutoff = config.new_word_cutoff
        self.num_workers = config.dataloader_num_workers
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.seed = config.seed
        self.world_size = config.world_size
        self.rank = config.rank
        self.llm = config.llm
        self.val_items = config.val_items
        logger.setLevel(self.logging_level)

        # Get a list of directories in data_path
        self.dirs = [x[0] for x in os.walk(self.data_path)]
        self.dirs = self.dirs[1:]
        self.json_folders = {}
        self.llms = []
        if fixed_llm is None:
            logger.info("Found {} directories in data path.".format(len(self.dirs)))
            for i, dir in enumerate(self.dirs):
                dir_name = dir.split("/")[-1]
                logger.debug("Loading data in directory: {}".format(dir_name))
                json_files = [x[2] for x in os.walk(dir)][0]
                json_files = [x for x in json_files if x.endswith(".json")]
                if len(json_files) == 0:
                    logger.warning(f"No json files found in directory {dir_name}. Continuing.")
                    continue
                else:
                    logger.debug("Found {} json files in directory.".format(len(json_files)))
                    self.json_folders[dir_name] = json_files
                    self.llms.append(dir_name)

            logger.info("Found {} LLMs to train.".format(len(self.llms)))
            logger.debug("LLM: {}".format(self.llms))
        else:
            self.llms = [fixed_llm]
            dir = os.path.join(self.data_path, fixed_llm)
            json_files = [x[2] for x in os.walk(dir)][0]
            json_files = [x for x in json_files if x.endswith(".json")]
            if len(json_files) == 0:
                logger.error(f"No json files found in directory {fixed_llm}.")
                raise AttributeError
            else:
                logger.debug("Found {} json files in directory.".format(len(json_files)))
                self.json_folders[fixed_llm] = json_files

    @log()
    def make_dataset(self, llm: str, val_items=None, streaming=True) -> DatasetDict:
        """
        This function creates the dataset for a single LLM.
        We use Huggingface Datasets to lazily load the json data.
        """

        if val_items is None:
            val_items = self.val_items
        logger.debug("Creating dataset for LLM: {}".format(llm))
        # Get the json files for the LLM
        try:
            json_files = self.json_folders[llm]
            logger.debug("Found {} json files for LLM.".format(len(json_files)))
            # Create a list of paths to the json files
            json_paths = [os.path.join(self.data_path, llm, x) for x in json_files]
        except KeyError:
            logger.warning("No json files found for LLM: {}".format(llm))
            return None
        except Exception as e:
            logger.error(f"Error reading json files for LLM: {llm}. Error: {e}")
            return None
        try:
            # Try loading metadata picle from folder
            metadata_path = os.path.join(self.data_path, llm, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.debug("Loaded metadata for LLM: {}".format(llm))
        except Exception as e:
            logger.warning(f"Error loading metadata for LLM: {llm}. Error: {e}")
            metadata = None
        dataset = load_dataset(
            "json", data_files=json_paths, cache_dir=self.cache_dir, streaming=streaming
        )
        logger.debug("Loaded dataset for LLM: {}".format(llm))

        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed)

        if val_items > 0:
            logger.debug("Splitting dataset for LLM: {}".format(llm))
            if streaming:
                train_dataset = dataset["train"].skip(val_items)
                val_dataset = dataset["train"].take(val_items)
            else:
                train_dataset = dataset["train"].train_test_split(
                    test_size=val_items, shuffle=False
                )["train"]
                val_dataset = dataset["train"].train_test_split(test_size=val_items, shuffle=False)[
                    "test"
                ]
            dataset["train"] = train_dataset
            dataset["validation"] = val_dataset
            logger.debug("Split dataset for LLM: {}".format(llm))
        return dataset

    @log()
    def compute_length(self, llm):
        dataset = self.make_dataset(llm=llm, val_items=0, streaming=True)
        return sum(1 for _ in dataset["train"])

    @log()
    def get_missing_tokens(self, llm_list: list[str], tokenizer: Tokenizer) -> pd.DataFrame:
        """Gets missing tokens for a list of llm to train
        and combines them into a single list, where frequency cutoff determines which to drop
        """

        freq_tables = []
        for llm in llm_list:
            freq_table = self._get_missing_freq_table(llm, tokenizer)
            if freq_table is not None:
                freq_tables.append(freq_table)
        if len(freq_tables) == 0:
            return []
        freq_table = pd.concat(freq_tables)
        freq_table.index.name = "token"
        freq_table = freq_table.groupby("token").sum()
        freq_table = freq_table.sort_values(by="count", ascending=False)
        freq_table = freq_table[freq_table["count"] >= self.new_word_cutoff]
        return freq_table

    @log()
    def _get_missing_freq_table(self, llm: str, tokenizer: Tokenizer) -> pd.DataFrame:
        """This function gets missing tokens and packs them into a frequency table.
        This is done using HF datasets and iterating over batches of the dataset recursively.
        It 'should scale'."""
        dataset = self.make_dataset(llm, val_items=0)

        if dataset is None:
            return None

        logger.debug("Getting missing tokens for LLM: {}".format(llm))
        missing_tokens = []
        freq_table = {}

        def apply_nltk_ops(row):
            """NLTK Tokenization, stopword removal and missing token detection"""
            nltk_tokens = nltk.word_tokenize(row["sentence"])
            nltk_tokens = [w.lower() for w in nltk_tokens if (w.isalpha() and len(w) > 3)]
            stop_words = set(stopwords.words("english"))
            nltk_tokens = [w for w in nltk_tokens if w not in stop_words]
            missing_tokens = [x for x in nltk_tokens if x not in tokenizer.vocab]
            row["missing_tokens"] = missing_tokens
            return row

        logger.debug("Applying NLTK ops for LLM: {}".format(llm))
        missing_dataset = dataset["train"].map(
            apply_nltk_ops,
            batched=False,
            remove_columns=["sentence"],
        )
        logger.debug(f"Filtering missing tokens for LLM: {llm}")
        missing_dataset = missing_dataset.filter(lambda x: len(x["missing_tokens"]) > 0)

        def collate_fn(batch):
            batch = np.concatenate([b["missing_tokens"] for b in batch])
            return batch

        freq_table = {}
        # Create Dataloader and use Tqdm for progress bar
        logger.debug(f"Creating dataloader for missing tokens in: {llm}")
        dataloader = torch.utils.data.DataLoader(
            missing_dataset, batch_size=10, collate_fn=collate_fn
        )
        dataloader = tqdm(dataloader)

        # Iterate over batches and combine frequency tables
        for batch in dataloader:
            for word in batch:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

        # Here we retain the index of the frequency table, which is the missing token
        # later we will sum by index to combine tables
        freq_table = pd.DataFrame.from_dict(freq_table, orient="index", columns=["count"])
        freq_table = freq_table.sort_values(by="count", ascending=False)
        return freq_table
