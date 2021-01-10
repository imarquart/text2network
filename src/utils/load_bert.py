import torch
import logging

from torch import nn
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import BertConfig
import os

from transformers import BertTokenizer, WordpieceTokenizer
from collections import OrderedDict


class CustomVocabBertTokenizer(BertTokenizer):
    """
    Source: Transformers Git
    Adds tokens to vocab thus solving performance issue.
    """
    def add_tokens(self, new_tokens):
        new_tokens = [token for token in new_tokens if not (token in self.vocab or token in self.all_special_tokens)]

        self.vocab = OrderedDict([
            *self.vocab.items(),
            *[
                (token, i + len(self.vocab))
                for i, token in enumerate(new_tokens)
            ]
        ])

        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

        return len(new_tokens)

class CustomBertForMaskedLM(BertForMaskedLM):
    """
    Source: Transformers Git
    Correctly changes embedding size instead of erroring out when adding custom tokens
    """
    def __init__(self, config):
        super().__init__(config)

    def resize_embedding_and_fc(self, new_num_tokens):
        # Change the FC
        old_fc = self.cls.predictions.decoder
        self.cls.predictions.decoder = self._get_resized_fc(old_fc, new_num_tokens)

        # Change the bias
        old_bias = self.cls.predictions.bias
        self.cls.predictions.bias = self._get_resized_bias(old_bias, new_num_tokens)

        # Change the embedding
        self.resize_token_embeddings(new_num_tokens)

    def _get_resized_bias(self, old_bias, new_num_tokens):
        old_num_tokens = old_bias.data.size()[0]
        if old_num_tokens == new_num_tokens:
            return old_bias

        # Create new biases
        new_bias = nn.Parameter(torch.zeros(new_num_tokens))
        new_bias.to(old_bias.device)

        # Copy from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
        return new_bias

    def _get_resized_fc(self, old_fc, new_num_tokens):

        old_num_tokens, old_embedding_dim = old_fc.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_fc

        # Create new weights
        new_fc = nn.Linear(in_features=old_embedding_dim, out_features=new_num_tokens)
        new_fc.to(old_fc.weight.device)

        # initialize all weights (in particular added tokens)
        self._init_weights(new_fc)

        # Copy from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_fc.weight.data[:num_tokens_to_copy, :] = old_fc.weight.data[:num_tokens_to_copy, :]
        return new_fc


logger = logging.getLogger(__name__)


def get_full_vocabulary(tokenizer):
    """
    Adds together vocab and added tokens
    Parameters
    ----------
    tokenizer - Pytorch Transformer Tokenizer

    Returns
    -------
    ids, tokens
    """
    added_ids=list(tokenizer.added_tokens_decoder.keys())
    added_tokens=list(tokenizer.added_tokens_decoder.values())
    ids=list(tokenizer.vocab.values())
    tokens = list(tokenizer.vocab.keys())
    return ids+added_ids, tokens+added_tokens



def get_bert_and_tokenizer(modelpath,load_local=False,attentions=True,hidden=False):
    priorlevel=logging.root.level
    logging.disable(logging.ERROR)
    if load_local==True:
        try:
            #tokenizer = BertTokenizer.from_pretrained(modelpath, do_lower_case=True)
            tokenizer = CustomVocabBertTokenizer.from_pretrained(modelpath, do_lower_case=True)


            #bert = BertForMaskedLM.from_pretrained(modelpath, output_attentions=attentions, output_hidden_states=hidden)
            bert = CustomBertForMaskedLM.from_pretrained(modelpath, output_attentions=attentions, output_hidden_states=hidden)
        except:
            output_vocab_file = os.path.join(modelpath, 'bert-base-uncased-vocab.txt')
            output_model_file = os.path.join(modelpath, 'bert-base-uncased-pytorch_model.bin')
            output_config_file = os.path.join(modelpath, 'bert-base-uncased-config.json')

            #tokenizer = BertTokenizer.from_pretrained(output_vocab_file)
            tokenizer = CustomVocabBertTokenizer.from_pretrained(output_vocab_file)
            config = BertConfig.from_json_file(output_config_file)
            config.output_attentions = attentions
            config.output_hidden_states = hidden
            #bert = BertForMaskedLM.from_pretrained(output_model_file, config=config)
            bert = CustomBertForMaskedLM.from_pretrained(output_model_file, config=config)
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
    logging.disable(priorlevel)
    return tokenizer, bert
