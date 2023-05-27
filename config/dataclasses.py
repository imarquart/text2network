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
    group_by_length: bool = True
    world_size: int = 1
    dataloader_num_workers: int = 0
    dataloader_drop_last: bool = True
    rank: int = 1
    seed: int = 42


@dataclass
class LLMConfig:
    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    llm: str | None = None
    resume_from_checkpoint: str | None = None
    model_output_folder: str = "trained_models"


@dataclass
class TrainingConfig:
    data: TrainDataConfig
    output_folder: str = "data/output/trained_models"
    llm: LLMConfig = LLMConfig()
    logging_folder: str = "logs/training"
    logging_level: int = logging.DEBUG
    other_loggers: int = logging.WARNING
    mlm_probability: float = 0.2
    max_seq_length: int = 40
    gpu_batch: int = 120
    epochs: int = 1000
    max_steps: int = 10000000
    num_train_epochs: int | None = None
    max_eval_steps: int = 1000
    warmup_steps: int = 0
    save_steps: int = 500000
    eval_steps: int = 500
    eval_loss_limit: float = 0.5
    loss_limit: float = 0.45
    pad_tokenizer: bool = True
    fp16: bool = False
    gradient_accumulation_steps: int = 1
