# TODO: Redo Comments

import torch
import numpy as np
import tables
import time

import networkx as nx
from NLP.utils.rowvec_tools import simple_norm, get_weighted_edgelist, calculate_cutoffs
from NLP.src.datasets.text_dataset import text_dataset,text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_tensor import get_bert_tensor
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.utils.delwords import create_stopword_list
import tqdm




def process_sentences_network(tokenizer, bert, text_db, tensor_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size,nr_workers=0,copysort=True,method="attention",filters = tables.Filters(complevel=9, complib='blosc'),ch_shape=None):
    """
    Extracts probability distributions from texts and saves them in pyTables database
    in three formats:

    Sequence Data: Tensor for each sequence including distribution for each word.

    Token-Sequence Data: One entry for each token in each sequence, including all distributions of
    the sequence (duplication). Allows indexing by token.

    Token Data: One entry for each token in each sequence, but only includes fields for
    distribution of focal token, and and aggregation (average weighted by attention) of all contextual tokens.

    :param tokenizer: BERT tokenizer (pyTorch)
    :param bert: BERT model
    :param text_db: HDF5 File of processes sentences, string of tokens, ending with punctuation
    :param tensor_db: HDF5 File to save processed tensors
    :param MAX_SEQ_LENGTH:  maximal length of sequences
    :param DICT_SIZE: tokenizer dict size
    :param batch_size: batch size to send to BERT
    :param nr_workers: Nr workers for dataloader. Probably should be set to 0 on windows
    :param copysort: At the end of the operation, sort table by token id, reapply compression and save (recommended)
    :param method: "attention": Weigh by BERT attention; "context_element": Sum probabilities unweighted
    :return: None
    """
    tables.set_blosc_max_threads(15)

    # %% Initialize text dataset
    dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader=DataLoaderX(dataset=dataset,batch_size=None, sampler=batch_sampler,num_workers=nr_workers,collate_fn=text_dataset_collate_batchsample, pin_memory=False)


    # Push BERT to GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # Create Stopwords
    delwords=create_stopword_list(tokenizer)
    # Create Graphs
    graph = nx.MultiDiGraph()
    context_graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(0, tokenizer.vocab_size))
    context_graph.add_nodes_from(range(0, tokenizer.vocab_size))

    # Counter for timing
    model_timings=[]
    process_timings=[]
    load_timings=[]
    start_time = time.time()
    for batch, seq_ids, token_ids in tqdm.tqdm(dataloader, desc="Iteration"):

        # Data spent on loading batch
        load_time=time.time()-start_time
        load_timings.append(load_time)
        # This seems to allow slightly higher batch sizes on my GPU
        #torch.cuda.empty_cache()
        # Run BERT and get predictions
        predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,return_max=False)

        # compute model timings time
        prepare_time=time.time()-start_time-load_time
        model_timings.append(prepare_time)

        # %% Sequence Table
        # Iterate over sequences
        for sequence_id in np.unique(seq_ids):
            sequence_mask = seq_ids == sequence_id
            sequence_size = sum(sequence_mask)

            # Pad and add sequence of IDs
            idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
            idx[0, :sequence_size] = token_ids[sequence_mask]
            # Pad and add distributions per token, we need to save to maximum sequence size
            dists = torch.zeros([MAX_SEQ_LENGTH, DICT_SIZE], requires_grad=False)
            dists[:sequence_size, :] = predictions[sequence_mask, :]

            # %% Token Table

            if method=="attention":
                # Extract attention for sequence
                seq_attn=attn[sequence_mask,:].cpu()
                # Curtail to tokens in sequence
                # attention row vectors for each token are of
                # size sequence_size+2, where position 0 is <CLS>
                # and position n+1 is <SEP>, these we ignore
                seq_attn=seq_attn[:,1:sequence_size+1]
                # Delete diagonal attention
                seq_attn[torch.eye(sequence_size).bool()]=0
                # Normalize
                #seq_attn=torch.div(seq_attn.transpose(-1, 0), torch.sum(seq_attn, dim=1)).transpose(-1, 0)
            else:
                # Context element distribution: we sum over all probabilities in a sequence
                seq_attn=torch.ones([sequence_size,sequence_size])
                seq_attn[torch.eye(sequence_size).bool()]=0


            for pos, token in enumerate(token_ids[sequence_mask]):
                # Add Network logic here
                if token.numpy() not in delwords:
                    replacement = dists[pos, :]
                    ## Context distribution with attention weights
                    context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
                    context_index[:sequence_size] = True
                    context_dist = dists[context_index, :]
                    context =(torch.sum((seq_attn[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0), dim=0).unsqueeze(0))

                    replacement = simple_norm(replacement.numpy())
                    context = simple_norm(context.numpy())
                    replacement=replacement.flatten()
                    context=context.flatten()
                    cutoff_number, cutoff_probability = calculate_cutoffs(replacement, method="percent", percent=80)
                    graph.add_weighted_edges_from(
                        get_weighted_edgelist(token, replacement, cutoff_number, cutoff_probability), 'weight',
                        seq_id=sequence_id, pos=pos)
                    cutoff_number, cutoff_probability = calculate_cutoffs(context, method="percent", percent=80)
                    graph.add_weighted_edges_from(
                        get_weighted_edgelist(token, context, cutoff_number, cutoff_probability), 'weight',
                        seq_id=sequence_id, pos=pos)

        del predictions, attn

        # compute processing time
        process_timings.append(time.time()-start_time - prepare_time - load_time)
        # New start time
        start_time=time.time()




    dataset.close()

    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Model Time: %s seconds" % (np.mean(model_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings)/np.mean(process_timings+model_timings)))

    return graph, context_graph
