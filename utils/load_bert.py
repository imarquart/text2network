import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import BertConfig
import os

def get_bert_and_tokenizer(modelpath):

    try:
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                               'bert-base-uncased')
        bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')
        if tokenizer is None or bert is None:
            raise ImportError
    except:
        try:
            print("Can not download BERT & Tokenizer, trying local")
            output_vocab_file=os.path.join(modelpath,'bert-base-uncased-vocab.txt')
            output_model_file = os.path.join(modelpath,'bert-base-uncased-pytorch_model.bin')
            output_config_file = os.path.join(modelpath,'bert-base-uncased-config.json')

            tokenizer = BertTokenizer.from_pretrained(output_vocab_file)
            config = BertConfig.from_json_file(output_config_file)
            bert = BertForMaskedLM.from_pretrained(output_model_file,config=config)
        except:
            raise ImportError

    return tokenizer,bert