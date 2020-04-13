# TODO: Redo Comments

import torch
import numpy as np
import tables
import time
import logging
import networkx as nx
from NLP.utils.rowvec_tools import simple_norm, add_to_networks
from NLP.src.datasets.text_dataset import text_dataset, text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_tensor import get_bert_tensor
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.src.utils.delwords import create_stopword_list
import tqdm


def process_sentences_network(tokenizer, bert, text_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size, nr_workers=0,
                              cutoff_percent=80,max_degree=100):
    """
    Extracts pre-processed sentences, gets predictions by BERT and creates a network

    Network is created for both context distribution and for replacement distribution
    Each sentence is added via parallel ties in a multigraph

    :param tokenizer: BERT tokenizer (pyTorch)
    :param bert: BERT model
    :param text_db: HDF5 File of processes sentences, string of tokens, ending with punctuation
    :param MAX_SEQ_LENGTH:  maximal length of sequences
    :param DICT_SIZE: tokenizer dict size
    :param batch_size: batch size to send to BERT
    :param nr_workers: Nr workers for dataloader. Probably should be set to 0 on windows
    :param method: "attention": Weigh by BERT attention; "context_element": Sum probabilities unweighted
    :param cutoff_percent: Amount of probability mass to use to create links. Smaller values, less ties.
    :return: graph, context_graph (networkx DiMultiGraphs)
    """
    tables.set_blosc_max_threads(15)

    # %% Initialize text dataset
    dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH)
    logging.info("Number of sentences found: %i"%dataset.nitems)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=nr_workers,
                             collate_fn=text_dataset_collate_batchsample, pin_memory=False)

    # Push BERT to GPU
    torch.cuda.empty_cache()
    if torch.cuda.is_available(): logging.info("Using CUDA.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # Create Stopwords
    delwords = create_stopword_list(tokenizer)
    # Create Graphs
    graph = nx.MultiDiGraph()
    context_graph = nx.MultiDiGraph()
    attention_graph = nx.MultiDiGraph()

    graph.add_nodes_from(range(0, tokenizer.vocab_size))
    context_graph.add_nodes_from(range(0, tokenizer.vocab_size))
    attention_graph.add_nodes_from(range(0, tokenizer.vocab_size))

    # Counter for timing
    model_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()
    for batch, seq_ids, token_ids in tqdm.tqdm(dataloader, desc="Iteration"):

        # Data spent on loading batch
        load_time = time.time() - start_time
        load_timings.append(load_time)
        # This seems to allow slightly higher batch sizes on my GPU
        # torch.cuda.empty_cache()
        # Run BERT and get predictions
        predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,
                                            return_max=False)

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
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


            #%% Extract attention for sequence
            # DISABLED ATTENTION
            #seq_attn = attn[sequence_mask, :].cpu()

            # Curtail to tokens in sequence
            # attention row vectors for each token are of
            # size sequence_size+2, where position 0 is <CLS>
            # and position n+1 is <SEP>, these we ignore

            # DISABLED ATTENTION
            #seq_attn = seq_attn[:, 1:sequence_size + 1]

            # Delete diagonal attention
            # DISABLED ATTENTION
            #seq_attn[torch.eye(sequence_size).bool()] = 0

            #%% Context element distribution: we sum over all probabilities in a sequence
            seq_ce = torch.ones([sequence_size, sequence_size])
            seq_ce[torch.eye(sequence_size).bool()] = 0

            # TODO: Re-Enable Lines of attention
            for pos, token in enumerate(token_ids[sequence_mask]):
                # Should all be np
                token=token.item()
                if token not in delwords:
                    replacement = dists[pos, :]

                    ## Context distributions
                    context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
                    context_index[:sequence_size] = True
                    context_dist = dists[context_index, :]

                    ## Attention
                    # DISABLED ATTENTION
                    #context_att = (
                    #    torch.sum((seq_attn[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0), dim=0).unsqueeze(0))

                    ## Context Element
                    context = (
                        torch.sum((seq_ce[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0), dim=0).unsqueeze(0))
                    # Flatten, since it is one row each
                    replacement = replacement.numpy().flatten()

                    # DISABLED ATTENTION
                    #context_att = context_att.numpy().flatten()

                    context = context.numpy().flatten()

                    # Sparsify
                    # TODO: try without setting own-link to zero!
                    replacement[token]=0
                    replacement[replacement==np.min(replacement)]=0
                    context[context==np.min(context)]=0

                    # DISABLED ATTENTION
                    # context_att[context_att==np.min(context_att)]=0

                    # Get rid of delnorm links
                    replacement[delwords]=0
                    context[delwords]=0

                    # DISABLED ATTENTION
                    # context_att[delwords]=0

                    # We norm the distributions here
                    replacement = simple_norm(replacement)
                    context = simple_norm(context)

                    # DISABLED ATTENTION
                    # context_att = simple_norm(context_att)

                    # Add values to network
                    graph,context_graph,attention_graph=add_to_networks(graph,context_graph,attention_graph,replacement,context,context,token,cutoff_percent,max_degree,pos,sequence_id)

            del dists

        del predictions, attn

        # compute processing time
        process_timings.append(time.time() - start_time - prepare_time - load_time)
        # New start time
        start_time = time.time()

    dataset.close()

    logging.info("Average Load Time: %s seconds" % (np.mean(load_timings)))
    logging.info("Average Model Time: %s seconds" % (np.mean(model_timings)))
    logging.info("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    logging.info("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))

    return graph, context_graph, attention_graph
