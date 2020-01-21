import torch
import logging
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import BertConfig
import os

logger = logging.getLogger(__name__)


def get_bert_and_tokenizer_local(modelpath):
    output_vocab_file = os.path.join(modelpath, 'bert-base-uncased-vocab.txt')
    output_model_file = os.path.join(modelpath, 'bert-base-uncased-pytorch_model.bin')
    output_config_file = os.path.join(modelpath, 'bert-base-uncased-config.json')

    tokenizer = BertTokenizer.from_pretrained(output_vocab_file)
    config = BertConfig.from_json_file(output_config_file)
    config.output_attentions = True
    bert = BertForMaskedLM.from_pretrained(output_model_file, config=config)
    return tokenizer, bert


def get_bert_and_tokenizer(modelpath,load_local=False,attentions=True,hidden=False):
    logger.setLevel(logging.WARN)
    if load_local==True:
        try:
            tokenizer = BertTokenizer.from_pretrained(modelpath, do_lower_case=True)
            bert = BertForMaskedLM.from_pretrained(modelpath, output_attentions=attentions, output_hidden_states=hidden)
        except:
            output_vocab_file = os.path.join(modelpath, 'bert-base-uncased-vocab.txt')
            output_model_file = os.path.join(modelpath, 'bert-base-uncased-pytorch_model.bin')
            output_config_file = os.path.join(modelpath, 'bert-base-uncased-config.json')

            tokenizer = BertTokenizer.from_pretrained(output_vocab_file)
            config = BertConfig.from_json_file(output_config_file)
            config.output_attentions = attentions
            config.output_hidden_states = hidden
            bert = BertForMaskedLM.from_pretrained(output_model_file, config=config)
    else:
        try:
            tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                                       'bert-base-uncased')
            bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased',
                                  output_attentions=attentions, output_hidden_states=hidden)
            if tokenizer is None or bert is None:
                raise ImportError
        except:
            raise ImportError
    logger.setLevel(logging.INFO)
    return tokenizer, bert
