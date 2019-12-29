# TODO: Redo Comments

import torch
import numpy as np
import tables
import time

from NLP.src.datasets.text_dataset import text_dataset, text_dataset_collate_batchsample
from NLP.src.datasets.dataloaderX import DataLoaderX
from NLP.src.text_processing.get_bert_tensor import get_bert_tensor
from torch.utils.data import BatchSampler, SequentialSampler
from NLP.utils.delwords import create_stopword_list
import tqdm


def process_sentences(tokenizer, bert, text_db, tensor_db, temp_db,MAX_SEQ_LENGTH, DICT_SIZE, batch_size, nr_workers=0,
                      copysort=True, method="attention", filters=tables.Filters(complevel=9, complib='blosc'),
                      ch_shape=None):
    """
    Extracts probability distributions from texts and saves them in pyTables database

    Token Data: One entry for each token in each sequence, but only includes fields for
    distribution of focal token, and and aggregation (average or weighted by attention) of all contextual tokens.

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

    # If not set
    tables.set_blosc_max_threads(15)

    # %% Initialize text dataset
    dataset = text_dataset(text_db, tokenizer, MAX_SEQ_LENGTH)
    batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=batch_size, drop_last=False)
    # We use a dataloader than can load samples threaded, it wraps the normal pytorch dataloader
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=nr_workers,
                             collate_fn=text_dataset_collate_batchsample, pin_memory=False)

    # Calculate expected rows, for optimization of pyTables
    expected_rows = np.int(dataset.nitems * 7.4)

    # Tables particle for each token occurrence
    class Token_Particle(tables.IsDescription):
        token_id = tables.UInt32Col()
        seq_id = tables.UInt32Col()
        pos_id = tables.UInt32Col()
        seq_size = tables.UInt32Col()
        own_dist = tables.Float32Col(shape=(1, DICT_SIZE))
        context_dist = tables.Float32Col(shape=(1, DICT_SIZE))


    # Open database storing token data
    try:
        data_file = tables.open_file(temp_db, mode="w", title="Data File", filters=filters)
        final_file = tables.open_file(tensor_db, mode="w", title="Data File", filters=filters)

    except:
        raise ConnectionError("Can not write tensor databse file")

    # We create two groups, as we will copy and re-sort the table
    group = data_file.create_group("/", 'token_data', 'Token Data')
    final_group = final_file.create_group("/", 'token_data', 'Token Data')
    token_table = data_file.create_table(group, 'table', Token_Particle, "Token Table", expectedrows=expected_rows,
                                         chunkshape=ch_shape)
    # Create index ONLY when not sorting later
    # We want to reindex otherwise, and enable autochunking etc.
    token_table.cols.token_id.create_csindex()

    # Push BERT to GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    # Ignore word-list
    delwords = create_stopword_list(tokenizer)

    # Counter for timing
    model_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()

    # Main loop over batches
    for batch, seq_ids, token_ids in tqdm.tqdm(dataloader, desc="Iteration"):

        # Data spent on loading batch
        load_time = time.time() - start_time
        load_timings.append(load_time)

        # Enable this to allow slightly higher batch sizes on GPU, may be slower
        # torch.cuda.empty_cache()
        # Run BERT and get predictions
        predictions, attn = get_bert_tensor(0, bert, batch, tokenizer.pad_token_id, tokenizer.mask_token_id, device,
                                            return_max=False)

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        model_timings.append(prepare_time)


        # Iterate over sequences
        for sequence_id in np.unique(seq_ids):

            # Each batch may include several sequences, select the current one
            sequence_mask = seq_ids == sequence_id
            sequence_size = sum(sequence_mask)

            # Contruct padded sequence of token id's
            idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
            idx[0, :sequence_size] = token_ids[sequence_mask]
            # Pad and add distributions per token, we need to save to maximum sequence size
            dists = torch.zeros([MAX_SEQ_LENGTH, DICT_SIZE], requires_grad=False)
            dists[:sequence_size, :] = predictions[sequence_mask, :]

            # Calculate weights of context tokens
            if method == "attention":
                # Extract attention for current sequence
                seq_attn = attn[sequence_mask, :].cpu()
                # Curtail to tokens in sequence
                # attention row vectors for each token are of
                # size sequence_size+2, where position 0 is <CLS>
                # and position n+1 is <SEP>, these we ignore
                seq_attn = seq_attn[:, 1:sequence_size + 1]
                # Delete diagonal attention
                seq_attn[torch.eye(sequence_size).bool()] = 0
                # We do not normalize here
                # seq_attn=torch.div(seq_attn.transpose(-1, 0), torch.sum(seq_attn, dim=1)).transpose(-1, 0)
            else:
                # Context element distribution: we sum over all probabilities in a sequence
                seq_attn = torch.ones([sequence_size, sequence_size])
                seq_attn[torch.eye(sequence_size).bool()] = 0

            for pos, token in enumerate(token_ids[sequence_mask]):
                if token.numpy() not in delwords:
                    particle = token_table.row
                    particle['token_id'] = token
                    particle['pos_id'] = pos
                    particle['seq_id'] = sequence_id
                    particle['seq_size'] = sequence_size
                    # Replacement distribution
                    particle['own_dist'] = dists[pos, :].unsqueeze(0)
                    # Create context distribution.
                    # We mask padding entries here
                    # Note that dists, as opposed to attention, does not have special tokens anymore
                    context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
                    context_index[:sequence_size] = True
                    # Select only context elements that are not padding
                    context_dist = dists[context_index, :]
                    # The context distribution corresponds to the (weighted) sum across sequence replacement vectors
                    particle['context_dist'] = (
                        torch.sum((seq_attn[pos] * context_dist.transpose(-1, 0)).transpose(-1, 0), dim=0).unsqueeze(0))

                    particle.append()

        # To free up memory on the GPU
        del predictions, attn

        # compute processing time
        process_timings.append(time.time() - start_time - prepare_time - load_time)
        # New start time
        start_time = time.time()

    data_file.flush()


    # Try to sort table
    # This is important for accessing by token later on
    if copysort == True:
        print("Copying table, reindexing, and sorting of %i rows." % token_table.nrows)
        # Save table name
        oldname = token_table.name
        # If there is no index, we need a CSI (full) index
        if token_table.cols.token_id.is_indexed == False:
            token_table.cols.token_id.create_csindex()
        else:
            token_table.cols.token_id.reindex()
        # Now we know the number of rows, so let autochunk use that value
        expected_rows = token_table.nrows
        # We copy the table into a new group
        newtable = token_table.copy(newparent=final_group, newname='sortedset', sortby='token_id', propindexes=True,
                                    filters=filters, chunkshape="auto", expected_rows=expected_rows)
        newtable.rename(oldname)
        final_file.flush()
        print("Removing old data")
        #token_table.remove()
        # Here we wish to delete the unsorted table. Sadly, pyTables does not seem to delete the data
        # or reduce file size for removing the table, so we remove the whole group node, having
        # placed the copy in a new group
        #group.remove()
        # Flush to apply changes
        data_file.flush()

    dataset.close()
    data_file.close()
    final_file.close()

    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Model Time: %s seconds" % (np.mean(model_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))
