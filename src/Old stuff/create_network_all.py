# TODO: Commment
# TODO: Add Logger

from NLP.src.datasets.dataloaderX import DataLoaderX
import numpy as np
import networkx as nx
from tqdm import tqdm
from NLP.src.utils.delwords import create_stopword_list
from torch.utils.data import BatchSampler, SequentialSampler
import time
from NLP.src.datasets.tensor_dataset import tensor_dataset, tensor_dataset_collate_batchsample
from NLP.utils.rowvec_tools import simple_norm, add_to_networks


def create_network_all(database, tokenizer, batch_size=0, dset_method=0, cutoff_percent=80):
    """
    If we have saved the replacement and context distributions in a hd5 file, we can create the network
    without re-running BERT.

    This function creates a multiplex (multigraph) network, where each occurrence forms a set of links

    :param database: The Tensor databse to read
    :param tokenizer: The BERT tokenizer
    :param batch_size: Size of batches to read from the database
    :param dset_method: Our dataset can either query single entries or multiple, albeit we see little timing difference
    :param cutoff_percent: The percentage of probability used to create network. Lower => sparser network
    :return: graph, context_graph - Graph of replacement and context network
    """
    # Initialize dataset
    dataset = tensor_dataset(database, method=dset_method)

    # Delete from this stopwords and the like
    # This is a safeguard and should not be necessary
    delwords = create_stopword_list(tokenizer)
    dataset.nodes = np.setdiff1d(dataset.nodes, delwords)
    nr_nodes = dataset.nodes.shape[0]

    # Create Multigraphs
    graph = nx.MultiDiGraph()
    context_graph = nx.MultiDiGraph()
    # Mainly to allow mapping from token-ids to token, we create tokens for every word in the dictionary
    # although our node list does not include stopwords
    graph.add_nodes_from(range(0, tokenizer.vocab_size))
    context_graph.add_nodes_from(range(0, tokenizer.vocab_size))

    # We use a threaded Dataloader, that wraps the pyTorch dataloader, batch_size is zero as batching is done in the dataset
    btsampler = BatchSampler(SequentialSampler(dataset.nodes), batch_size=batch_size, drop_last=False)
    dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=btsampler, num_workers=0,
                             collate_fn=tensor_dataset_collate_batchsample, pin_memory=False)

    # Counter for timing
    prep_timings = []
    process_timings = []
    load_timings = []
    start_time = time.time()

    # A batch element are all occurrences of a single token
    # If the batch size is larger than 1, several tokens are queried
    for chunk, token_idx, own_dists, context_dists in tqdm(dataloader):
        # Data spent on loading batch
        load_time = time.time() - start_time
        load_timings.append(load_time)

        # The number of occurrences within our samples (possibly across several tokens if batch size >1)
        nr_rows = len(token_idx)
        if nr_rows == 0:
            raise AssertionError("Database error: Token information missing")

        # Set probabilities of stopwords to zero
        context_dists[:, delwords] = 0  # np.min(context_dists)
        own_dists[:, delwords] = 0

        # Find a better way to do this
        # thsi is terrible
        for token in chunk:
            context_dists[token_idx == token, token] = 0
            own_dists[token_idx == token, token] = 0

        # compute model timings time
        prepare_time = time.time() - start_time - load_time
        prep_timings.append(prepare_time)

        # Loop over each token
        for token in chunk:
            # There are likely several occurrences for each token
            mask = (token_idx == token)
            if len(token_idx[mask]) > 0:

                # Subset the two distributions
                replacement = own_dists[mask, :]
                context = context_dists[mask, :]

                # Here we proceed for each occurrence
                # TODO: Vectorize this for speed
                for i in range(0, len(token_idx[mask])):
                    # Due to us setting some values to zero, we need to re-norm
                    replacement[i, :] = simple_norm(replacement[i, :])
                    context[i, :] = simple_norm(context[i, :])

                    # Add to the networks
                    graph, context_graph = add_to_networks(graph, context_graph, replacement[i, :], context[i, :],
                                                           token, cutoff_percent, 0, i)

        process_timings.append(time.time() - start_time - prepare_time - load_time)
    print(" ")
    print("Average Load Time: %s seconds" % (np.mean(load_timings)))
    print("Average Prep Time: %s seconds" % (np.mean(prep_timings)))
    print("Average Processing Time: %s seconds" % (np.mean(process_timings)))
    print("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + prep_timings)))
    return graph, context_graph
