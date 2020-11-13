year = 2019
sents_to_extract=[2181,4809,12913,6516,9060,13233,1396,869]
cutoff_number=20



import pandas as pd
import numpy as np
import tables
import logging
from NLP.utils.rowvec_tools import simple_norm
from NLP.src.datasets.text_dataset import text_dataset, text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_tensor import get_bert_tensor
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.src.utils.delwords import create_stopword_list
import torch
from NLP.config.config import configuration
from NLP.utils.load_bert import get_bert_and_tokenizer

# %% Config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
cfg = configuration()
# %% Create folder structure
logging.info("Setting up folder structure")
text_folder = ''.join([cfg.text_folder, '/', str(year)])
data_folder = ''.join([cfg.data_folder, '/', str(year)])
bert_folder = ''.join([data_folder, '/bert'])
cluster_xls = ''.join([cfg.data_folder, '/cluster_xls'])

text_db = ''.join([text_folder, '/', str(year), '.h5'])

tokenizer, bert = get_bert_and_tokenizer(bert_folder, True)
delwords = create_stopword_list(tokenizer)
DICT_SIZE = tokenizer.vocab_size
MAX_SEQ_LENGTH=30
tables.set_blosc_max_threads(15)
batch_size=30
nr_workers=0

# %% Initialize text dataset
dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH)
logging.info("Number of sentences found: %i" % dataset.nitems)
batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=nr_workers,
                         collate_fn=text_dataset_collate_batchsample, pin_memory=False)

# Push BERT to GPU
torch.cuda.empty_cache()
if torch.cuda.is_available(): logging.info("Using CUDA.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert.to(device)
bert.eval()

list_seq=[]
list_pos=[]
list_ftoken=[]
list_token_id=[]
list_weight=[]
list_rank=[]
for seq in sents_to_extract:
    batch, seq_ids, token_ids=dataset[seq]
    sequence_id=seq
    predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,
                                        return_max=False)
    sequence_mask = seq_ids == sequence_id
    sequence_size = sum(sequence_mask)

    # Pad and add sequence of IDs
    idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
    idx[0, :sequence_size] = token_ids[sequence_mask]
    # Pad and add distributions per token, we need to save to maximum sequence size
    dists = torch.zeros([MAX_SEQ_LENGTH, DICT_SIZE], requires_grad=False)
    dists[:sequence_size, :] = predictions[sequence_mask, :]

    for pos, token in enumerate(token_ids[sequence_mask]):
        replacement = dists[pos, :]
        replacement = replacement.numpy().flatten()
        replacement[token] = 0
        replacement[replacement == np.min(replacement)] = 0
        replacement = simple_norm(replacement)
        neighbors = np.argsort(-replacement)[:cutoff_number]
        add_weights = replacement[neighbors]
        add_neighbors=tokenizer.convert_ids_to_tokens(neighbors)
        add_seq=np.repeat(sequence_id,len(add_weights))
        ftoken=np.repeat(token,len(add_weights)).numpy()
        add_ftoken=tokenizer.convert_ids_to_tokens(ftoken)
        add_pos=np.repeat(pos,len(add_weights))
        add_rank=list(range(0,len(add_neighbors)))
        # Add to lists
        list_ftoken.extend(add_ftoken)
        list_token_id.extend(add_neighbors)
        list_weight.extend(add_weights)
        list_pos.extend(add_pos)
        list_seq.extend(add_seq)
        list_rank.extend(add_rank)
        # Del to make sure no mix ups
        del add_weights, add_neighbors, add_ftoken, add_pos, add_seq

# Create dict
data_dict={"Seq":list_seq, "Pos": list_pos, "FocalToken": list_ftoken, "RepToken":list_token_id, "Prob":list_weight, "Rank":list_rank}

# Create DataFrame
data_table=pd.DataFrame(data_dict)
fname_xls="".join([cluster_xls,"/Replacement_extracts.xlsx"])
fname_csv="".join([cluster_xls,"/Replacement_extracts.csv"])

data_table.to_excel(fname_xls)
data_table.to_csv(fname_csv)


for i in sents_to_extract:
    a=tokenizer.convert_ids_to_tokens(dataset[i][0].numpy().flatten())
    print(a)
    #if "leader" in a:
    #    if ("innovate" in a) or ("innovation" in a) or ("ai" in a) or ("digital" in a) or ("transformation" in a) or ("transformative" in a):
    #        print(i)



