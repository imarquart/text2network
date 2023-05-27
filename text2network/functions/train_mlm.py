import copy
import logging
import math
import os
from dataclasses import dataclass

import evaluate
from accelerate import Accelerator
from datasets import Dataset, load_dataset
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
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from config.dataclasses import TrainDataConfig, TrainingConfig
from text2network.datasets.text_datasets import PreTrainDataSet
from text2network.utils.file_helpers import check_create_folder

logger = logging.getLogger("t2n")


def train_mlm_simple(config: TrainingConfig):
    """_summary_

    Args:
        config: _description_
    """

    set_seed(config.data.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.tokenizer_name_or_path
        if config.llm.tokenizer_name_or_path is not None
        else config.llm.model_name_or_path
    )
    model = AutoModelForMaskedLM.from_pretrained(config.llm.model_name_or_path)
    # Dataset class
    pts = PreTrainDataSet(config.data, fixed_llm=config.llm.llm)
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = not config.pad_tokenizer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )
    # Check epoch config
    if config.num_train_epochs is not None:
        logger.info(f"Number of epochs given. Computing Dataset length")
        len_dataset = pts.compute_length(llm=config.llm.llm)
        max_examples = config.num_train_epochs * len_dataset
        assert max_examples > 1
        max_steps = max_examples // config.gpu_batch
        config.max_steps = max_steps
        logger.info(
            f"Dataset length {len_dataset} for {config.num_train_epochs} epochs "
            f"and batch_size {config.gpu_batch} implies {max_steps} steps."
        )

    config.output_folder = check_create_folder(config.output_folder)

    def tokenize_function(examples):
        result = tokenizer(
            examples["sentence"],
            padding="max_length" if config.pad_tokenizer else False,
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

    datasets = pts.make_dataset(llm=config.llm.llm)
    train_dataset = datasets["train"].with_format("torch")
    eval_dataset = datasets["validation"].with_format("torch")

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, batch_size=config.data.shuffle_buffer_size
    ).remove_columns(["sentence"])

    eval_dataset = eval_dataset.map(
        tokenize_function, batched=True, batch_size=config.data.shuffle_buffer_size
    ).remove_columns(["sentence"])

    training_args = TrainingArguments(
        output_dir=os.path.join(config.llm.model_output_folder, "checkpoints/"),
        logging_dir=os.path.join(config.llm.model_output_folder, "logs/"),
        logging_strategy="steps",
        logging_steps=config.eval_steps,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        seed=config.data.seed,
        save_steps=config.save_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.gpu_batch,
        per_device_eval_batch_size=config.gpu_batch,
        dataloader_drop_last=config.data.dataloader_drop_last,
        dataloader_num_workers=config.data.dataloader_num_workers,
        max_steps=config.max_steps,
    )

    # Cut dataset size
    if isinstance(train_dataset, Dataset):
        train_dataset = train_dataset.select(range(config.max_steps))
        eval_dataset = eval_dataset.select(range(config.max_eval_steps))
    else:
        train_dataset = train_dataset.take(config.max_steps)
        eval_dataset = eval_dataset.take(config.max_eval_steps)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    from transformers import TrainerCallback

    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer

        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = copy.deepcopy(control)
                self._trainer.evaluate(
                    eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
                )
                return control_copy

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if not is_torch_tpu_available()
        else None,
    )

    # trainer.add_callback(CustomCallback(trainer))

    # Training
    checkpoint = None
    if config.llm.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(config.llm.resume_from_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(
        output_dir=check_create_folder(config.llm.model_output_folder, "final/")
    )  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    metrics["train_samples"] = config.max_steps

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation

    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": config.llm.model_name_or_path, "tasks": "fill-mask"}
    kwargs["dataset"] = config.llm.llm

    trainer.create_model_card(**kwargs)


# def train_mlm_dist(config: TrainingConfig):
#     """_summary_

#     Args:
#         train_config: _description_
#         model: _description_
#         tokenizer: _description_
#         dataset: _description_


#         In a distributed setup like PyTorch DDP with a PyTorch DataLoader and shuffling
#         ```python
#         >>> from datasets.distributed import split_dataset_by_node
#         >>> ids = ds.to_iterable_dataset(num_shards=512)
#         >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
#         >>> ids = split_dataset_by_node(ds, world_size=8, rank=0)  # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
#         >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from this node's list of shards to each worker when you start iterating
#         >>> for example in ids:
#         ...     pass
#         ```

#         With shuffling and multiple epochs:
#         ```python
#         >>> ids = ds.to_iterable_dataset(num_shards=64)
#         >>> ids = ids.shuffle(buffer_size=10_000, seed=42)  # will shuffle the shards order and use a shuffle buffer when you start iterating
#         >>> for epoch in range(n_epochs):
#         ...     ids.set_epoch(epoch)  # will use effective_seed = seed + epoch to shuffle the shards and for the shuffle buffer when you start iterating
#         ...     for example in ids:
#         ...         pass
#         ```
#         Feel free to also use [`IterableDataset.set_epoch`] when using a PyTorch DataLoader or in distributed setups.
#     """

#     def tokenize_function(examples):
#         result = tokenizer(
#             examples["sentence"],
#             padding="max_length",
#             truncation=True,
#             max_length=config.max_seq_length,
#             return_tensors="pt",
#             # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
#             # receives the `special_tokens_mask`.
#             return_special_tokens_mask=True,
#         )
#         if tokenizer.is_fast:
#             result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

#         return result

#     config.data.world_size = accelerator.world_size
#     config.data.rank = accelerator.rank

#     pts = PreTrainDataSet(config.data)

#     with accelerator.main_process_first():
#         datasets = pts.make_dataset(llm=config.llm)
#         train_dataset = datasets["train"].with_format("torch")
#         val_dataset = datasets["validation"].with_format("torch")
#         train_dataset = split_dataset_by_node(
#             train_dataset, rank=config.data.rank, world_size=config.data.world_size
#         )
