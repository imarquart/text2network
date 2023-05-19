import logging
from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class PreProcessConfig:
    maximum_sequence_length: int = 128  # Max length of sentence
    split_symbol: str = "_"
    logging_level: int = logging.INFO
    other_loggers: int = logging.WARNING
    max_json_length: int = 20000  # Max length of json file in terms of rows
    input_folder: str = "data/raw"
    output_folder: str = "data/preprocessed"


@dataclass
class TrainDataConfig:
    data_path: str
    cache_dir: str = "cache"
    val_items: int = 1000
    llm: str | None = None
    new_word_cutoff: int = 10
    shuffle_buffer_size: int = 10_000
    world_size: int = 1
    dataloader_num_workers: int = 1
    rank: int = -1
    seed: int = 42


@dataclass
class LLMConfig:
    model_name_or_path: None | str = None
    llm: str| None= None


@dataclass
class TrainingConfig:
    data: TrainDataConfig
    model_path: str
    output_folder: str
    llm: LLMConfig | None = None
    trainer_args: TrainingArguments | None = None
    logging_folder: str = "logs/training"
    logging_level: int = 20
    other_loggers: int = 30
    mlm_probability: float = 0.2
    max_seq_length: int = 40
    gpu_batch: int = 120
    num_workers: int = 4
    epochs: int = 1000
    warmup_steps: int = 0
    save_steps: int = 500000
    eval_steps: int = 500
    eval_loss_limit: float = 0.5
    loss_limit: float = 0.45
