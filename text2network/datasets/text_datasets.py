import numpy as np
from tokenizers import Tokenizer
import torch
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numbers
import logging
import os
import pickle
import nltk
from nltk.tag import pos_tag, map_tag
from tqdm.auto import tqdm
from dataclasses import dataclass
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import log, setup_logger
from datasets import load_dataset, DatasetDict, Dataset
from config.dataclasses import TrainingConfig

nltk.download("stopwords")


@dataclass
class FakeTokenizer:
    vocab = []


# Setup logging
logger = logging.getLogger("t2n")


class PreTrainDataSet(object):
    """
    This dataset loads the data before training and determines two things:
    - The vocabulary to be added to the default LLM vocabulary
    - The split hierarchy and number of LLMs that need to be trained
    """

    @log(logging_level=20, other_loggers=30)
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_path = config.data_path
        self.output_folder = config.output_folder
        self.logging_folder = config.logging_folder
        self.logging_level = config.logging_level
        self.cache_dir = config.cache_dir
        self.other_loggers = config.other_loggers
        self.val_split = config.val_split
        self.new_word_cutoff = config.new_word_cutoff
        self.num_workers = config.num_workers
        self.sequence_batch = config.sequence_batch
        logger.setLevel(self.logging_level)

        # Get a list of directories in data_path
        self.dirs = [x[0] for x in os.walk(self.data_path)]
        self.dirs = self.dirs[1:]
        self.json_folders = {}
        self.llms = []
        logger.info("Found {} directories in data path.".format(len(self.dirs)))
        for i, dir in enumerate(self.dirs):
            dir_name = dir.split("/")[-1]
            logger.debug("Loading data in directory: {}".format(dir_name))
            json_files = [x[2] for x in os.walk(dir)][0]
            json_files = [x for x in json_files if x.endswith(".json")]
            if len(json_files) == 0:
                logger.warning(
                    f"No json files found in directory {dir_name}. Continuing."
                )
                continue
            else:
                logger.debug(
                    "Found {} json files in directory.".format(len(json_files))
                )
                self.json_folders[dir_name] = json_files
                self.llms.append(dir_name)

        logger.info("Found {} LLMs to train.".format(len(self.llms)))
        logger.debug("LLM: {}".format(self.llms))

    @log()
    def make_dataset(self, llm: str, val_split=None) -> DatasetDict:
        """
        This function creates the dataset for a single LLM.
        We use Huggingface Datasets to lazily load the json data.
        """
        if val_split is None:
            val_split = self.val_split
        logger.debug("Creating dataset for LLM: {}".format(llm))
        # Get the json files for the LLM
        try:
            json_files = self.json_folders[llm]
            logger.debug("Found {} json files for LLM.".format(len(json_files)))
            # Create a list of paths to the json files
            json_paths = [
                os.path.join(self.data_path, llm, x) for x in json_files
            ]
        except KeyError:
            logger.warning("No json files found for LLM: {}".format(llm))
            return None
        except Exception as e:
            logger.error(f"Error reading json files for LLM: {llm}. Error: {e}")
            return None
        dataset = load_dataset(
            "json", data_files=json_paths, cache_dir=self.cache_dir
        )
        logger.debug("Loaded dataset for LLM: {}".format(llm))
        if val_split > 0:
            logger.debug("Splitting dataset for LLM: {}".format(llm))
            dataset = dataset["train"].train_test_split(
                test_size=self.val_split
            )
            logger.debug("Split dataset for LLM: {}".format(llm))
        return dataset

    @log()
    def get_missing_freq_table(
        self, llm: str, tokenizer: Tokenizer
    ) -> pd.DataFrame:
        """This function gets missing tokens and packs them into a frequency table.
        This is done using HF datasets and iterating over batches of the dataset recursively.
        It 'should scale'."""
        dataset = self.make_dataset(llm, val_split=0)

        if dataset is None:
            return None

        logger.debug("Getting missing tokens for LLM: {}".format(llm))
        missing_tokens = []
        freq_table = {}

        def apply_nltk_ops(row):
            """NLTK Tokenization, stopword removal and missing token detection"""
            nltk_tokens = nltk.word_tokenize(row["sentence"])
            nltk_tokens = [
                w.lower() for w in nltk_tokens if (w.isalpha() and len(w) > 3)
            ]
            stop_words = set(stopwords.words("english"))
            nltk_tokens = [w for w in nltk_tokens if not w in stop_words]
            missing_tokens = [
                x for x in nltk_tokens if x not in tokenizer.vocab
            ]
            row["missing_tokens"] = missing_tokens
            return row

        def batched_freq_table(batch):
            """Given a batch of rows with missing tokens, return a frequency table"""
            freq_table = {}
            for row in batch["missing_tokens"]:
                text = row
                for word in text:
                    if word in freq_table:
                        freq_table[word] += 1
                    else:
                        freq_table[word] = 1
            pd_table = pd.DataFrame.from_dict(
                freq_table, orient="index", columns=["count"]
            ).reset_index()
            return {"freq_table": [pd_table]}

        def combine_two_freq_tables(table1, table2):
            """Combine two frequency tables"""
            table1 = dict(zip(table1["index"], table1["count"]))
            table2 = dict(zip(table2["index"], table2["count"]))
            table1.update(
                {
                    key: table2[key] + table1[key]
                    if key in table1
                    else table2[key]
                    for key in table2
                }
            )
            return table1

        def combine_batch_of_freq_tables(batch):
            """Combine a batch of frequency tables"""
            freq_table = {"count": [], "index": []}
            for table in batch["freq_table"]:
                freq_table = combine_two_freq_tables(freq_table, table)
                freq_table = {
                    "index": list(freq_table.keys()),
                    "count": list(freq_table.values()),
                }
            freq_table = dict(zip(freq_table["index"], freq_table["count"]))
            pd_table = pd.DataFrame.from_dict(
                freq_table, orient="index", columns=["count"]
            ).reset_index()
            return {"freq_table": [pd_table]}

        logger.debug("Applying NLTK ops for LLM: {}".format(llm))
        missing_dataset = dataset["train"].map(
            apply_nltk_ops,
            batched=False,
            remove_columns=["sentence"],
            num_proc=self.num_workers,
        )
        logger.debug(f"Filtering missing tokens for LLM: {llm}")
        missing_dataset = missing_dataset.filter(
            lambda x: len(x["missing_tokens"]) > 0, num_proc=self.num_workers
        )
        logger.debug(f"Mapping batched freq table for LLM: {llm}")
        batched_ds = missing_dataset.map(
            batched_freq_table,
            batched=True,
            batch_size=self.sequence_batch,
            remove_columns=missing_dataset.column_names,
            num_proc=self.num_workers,
        )

        logger.debug(f"Combining freq tables for LLM: {llm}")
        while len(batched_ds) > 1:
            batched_ds = batched_ds.map(
                combine_batch_of_freq_tables,
                batched=True,
                batch_size=self.sequence_batch,
            )

        freq_table = batched_ds["freq_table"][0]
        freq_table = dict(zip(freq_table["index"], freq_table["count"]))
        # Here we retain the index of the frequency table, which is the missing token
        # later we will sum by index to combine tables
        freq_table = pd.DataFrame.from_dict(
            freq_table, orient="index", columns=["count"]
        )
        freq_table = freq_table.sort_values(by="count", ascending=False)
        return freq_table

    @log()
    def get_missing_tokens(
        self, llm_list: list[str], tokenizer: Tokenizer
    ) -> list:
        """Gets missing tokens for a list of llm to train
        and combines them into a single list, where frequency cutoff determines which to drop
        """

        freq_tables = []
        for llm in llm_list:
            freq_table = self.get_missing_freq_table(llm, tokenizer)
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
