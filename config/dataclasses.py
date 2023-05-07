from dataclasses import dataclass


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
class TrainingConfig:
    data_path: str
    output_folder: str
    logging_folder: str = "logs/training"
    cache_dir: str = "cache"
    val_split: float = 0.1
    logging_level: int = 20
    other_loggers: int = 30
    new_word_cutoff: int = 10
    mlm_probability: float = 0.2
    max_seq_length: int = 40
    gpu_batch: int = 120
    sequence_batch: int = 10000
    num_workers: int = 4
    epochs: int = 1000
    warmup_steps: int = 0
    save_steps: int = 500000
    eval_steps: int = 500
    eval_loss_limit: int = 0.5
    loss_limit: int = 0.45
