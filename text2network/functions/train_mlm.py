from dataclasses import dataclass

from accelerate import Accelerator
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from config.dataclasses import TrainDataConfig, TrainingConfig
from text2network.datasets.text_datasets import PreTrainDataSet


def train_mlm_simple(config: TrainingConfig):
    """_summary_

    Args:
        config: _description_
    """

    tokenizer = AutoTokenizer.from_pretrained(config.llm.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(config.llm.model_name_or_path)
    datacollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )

    def tokenize_function(examples):
        result = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

        return result

    pts = PreTrainDataSet(config.data)

    datasets = pts.make_dataset(llm=config.llm)
    train_dataset = datasets["train"].with_format("torch")
    val_dataset = datasets["validation"].with_format("torch")

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, batch_size=config.data.shuffle_buffer_size
    ).remove_columns(["sentence"])


def train_mlm_dist(config: TrainingConfig):
    """_summary_

    Args:
        train_config: _description_
        model: _description_
        tokenizer: _description_
        dataset: _description_


        In a distributed setup like PyTorch DDP with a PyTorch DataLoader and shuffling
        ```python
        >>> from datasets.distributed import split_dataset_by_node
        >>> ids = ds.to_iterable_dataset(num_shards=512)
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> ids = split_dataset_by_node(ds, world_size=8, rank=0)  # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from this node's list of shards to each worker when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With shuffling and multiple epochs:
        ```python
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.shuffle(buffer_size=10_000, seed=42)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> for epoch in range(n_epochs):
        ...     ids.set_epoch(epoch)  # will use effective_seed = seed + epoch to shuffle the shards and for the shuffle buffer when you start iterating
        ...     for example in ids:
        ...         pass
        ```
        Feel free to also use [`IterableDataset.set_epoch`] when using a PyTorch DataLoader or in distributed setups.
    """

    def tokenize_function(examples):
        result = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

        return result

    config.data.world_size = accelerator.world_size
    config.data.rank = accelerator.rank

    pts = PreTrainDataSet(config.data)

    with accelerator.main_process_first():
        datasets = pts.make_dataset(llm=config.llm)
        train_dataset = datasets["train"].with_format("torch")
        val_dataset = datasets["validation"].with_format("torch")
        train_dataset = split_dataset_by_node(
            train_dataset, rank=config.data.rank, world_size=config.data.world_size
        )
