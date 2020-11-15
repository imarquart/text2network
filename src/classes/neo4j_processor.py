
# TODO: Redo Comments
# TODO: Add parameter handling!

import logging
import time
import numpy as np
import tables
import torch
import tqdm
from torch.utils.data import BatchSampler, SequentialSampler
from src.datasets.dataloaderX import DataLoaderX
from src.datasets.text_dataset import text_dataset, text_dataset_collate_batchsample
from src.utils.delwords import create_stopword_list
from src.utils.rowvec_tools import simple_norm

class neo4j_processor():
    def __init__(self, tokenizer, bert, neograph, MAX_SEQ_LENGTH, DICT_SIZE, batch_size,text_db=None,  maxn=None,nr_workers=0,cutoff_percent=99, max_degree=50,logging_level=logging.NOTSET):
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
        """
        # Assign parameters
        self.tokenizer=tokenizer
        self.bert=bert
        self.text_db=text_db
        self.neograph=neograph
        self.MAX_SEQ_LENGTH=MAX_SEQ_LENGTH
        self.DICT_SIZE=DICT_SIZE
        self.batch_size=batch_size
        self.maxn=maxn
        self.nr_workers=nr_workers
        self.cutoff_percent=cutoff_percent
        self.max_degree=max_degree
        self.logging_level=logging_level
        # Set logging level
        logging.disable(logging_level)

    def process_sentences_neo4j(self,year,text_db=None):
        """
        Extracts pre-processed sentences, gets predictions by BERT and creates network ties
        Ties are then added to the neo4j database.

        Network is created for both context distribution and for replacement distribution
        Each sentence is added via parallel ties in a multigraph

        :param year: year corresponding to variable in preprocessing database
        """
        tables.set_blosc_max_threads(15)


        # Check if text database has changed
        if text_db is not None:
            self.text_db=text_db
        elif self.text_db==None:
            logging.error("No text database provided!")
            raise ConnectionError("No text database provided!")

        # %% Initialize text dataset
        dataset = text_dataset(self.text_db, self.tokenizer, self.MAX_SEQ_LENGTH,maxn=self.maxn)
        logging.info("Number of sentences found: %i"%dataset.nitems)
        batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=self.batch_size, drop_last=False)
        dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=self.nr_workers,
                                collate_fn=text_dataset_collate_batchsample, pin_memory=False)

        # Push BERT to GPU
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): logging.info("Using CUDA.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)
        self.bert.eval()

        # Create Stopwords
        delwords = create_stopword_list(tokenizer)

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
            predictions, attn = self.get_bert_tensor(0, self.bert, batch, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, device,
                                                return_max=False)

            # compute model timings time
            prepare_time = time.time() - start_time - load_time
            model_timings.append(prepare_time)

            # %% Sequence Table
            # Iterate over sequences
            for sequence_id in np.unique(seq_ids):
                # Extract only current sequence
                sequence_mask = seq_ids == sequence_id
                sequence_size = sum(sequence_mask)

                # Pad and add sequence of IDs
                # TODO: Sanity check I don't need this anymore
                # idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
                # idx[0, :sequence_size] = token_ids[sequence_mask]
                # Pad and add distributions per token, we need to save to maximum sequence size
                dists = torch.zeros([MAX_SEQ_LENGTH, DICT_SIZE], requires_grad=False)
                dists[:sequence_size, :] = predictions[sequence_mask, :]

                # Context element selection matrix
                seq_ce = torch.ones([sequence_size, sequence_size])
                # Delete position of focal token
                seq_ce[torch.eye(sequence_size).bool()] = 0
                # Context distributions
                context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
                # Index words that are in the sentence
                context_index[:sequence_size] = True
                # Populate distribution with distributions
                # context_dist = dists[context_index, :]

                # TODO: This should probably not be a loop
                # but done smartly with matrices and so forth
                for pos, token in enumerate(token_ids[sequence_mask]):
                    # Should all be np
                    token = token.item()
                    if token not in delwords:

                        # %% Replacement distribution
                        replacement = dists[pos, :]
                        # Flatten, since it is one row each
                        replacement = replacement.numpy().flatten()
                        # Sparsify
                        # TODO: try without setting own-link to zero!
                        replacement[token] = 0
                        replacement[replacement == np.min(replacement)] = 0
                        # Get rid of delnorm links
                        replacement[delwords] = 0
                        # We norm the distributions here
                        replacement = simple_norm(replacement)

                        # %% Context Element
                        context = (
                            torch.sum((seq_ce[pos] * dists[context_index, :].transpose(-1, 0)).transpose(-1, 0),
                                    dim=0).unsqueeze(0))
                        # Flatten, since it is one row each
                        context = context.numpy().flatten()
                        # Sparsify
                        # TODO: try without setting own-link to zero!
                        context[token] = 0
                        context[context == np.min(context)] = 0
                        # Get rid of delnorm links
                        context[delwords] = 0
                        # We norm the distributions here
                        context = simple_norm(context)

                        # Add values to network
                        # Replacement ties
                        cutoff_number, cutoff_probability = self.calculate_cutoffs(replacement, method="percent",
                                                                            percent=cutoff_percent, max_degree=max_degree)
                        ties = self.get_weighted_edgelist(token, replacement, year, cutoff_number, cutoff_probability,
                                                    sequence_id, pos, max_degree=max_degree)

                        # Context ties
                        cutoff_number, cutoff_probability = self.calculate_cutoffs(context, method="percent",
                                                                            percent=cutoff_percent, max_degree=max_degree)
                        context_ties = self.get_weighted_edgelist(token, context, year, cutoff_number, cutoff_probability,
                                                            sequence_id, pos, max_degree=max_degree)

                        if (ties is not None) and (context_ties is not None):
                            neograph.insert_edges_context(token, ties, context_ties)


                del dists

            del predictions #, attn, token_ids, seq_ids, batch
            # compute processing time
            process_timings.append(time.time() - start_time - prepare_time - load_time)
            # New start time
            start_time = time.time()


        # Write remaining
        neograph.db.write_queue()
        torch.cuda.empty_cache()

        dataset.close()
        del dataloader, dataset, batch_sampler
        logging.info("Average Load Time: %s seconds" % (np.mean(load_timings)))
        logging.info("Average Model Time: %s seconds" % (np.mean(model_timings)))
        logging.info("Average Processing Time: %s seconds" % (np.mean(process_timings)))
        logging.info("Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))

    def get_bert_tensor(args, bert,tokens,pad_token_id,mask_token_id,device=torch.device("cpu"),return_max=False):
        """
        Extracts tensors of probability distributions for each word in sentence from BERT.
        This is done by running BERT separately for each token, masking the focal token.

        :param args: future: config
        :param bert: BERT model
        :param tokens: tensor of sequences
        :param pad_token_id: Token id's from tokenizer
        :param mask_token_id: Token id's from tokenizer
        :param device: CPU or CUDA device
        :param return_max: only returns ID of most likely token
        :return: predictions: Tensor of logits for each token (dimension: sum(k_i)*vocab-length); attn: Attention weights for each token
        """

        # We use lists of tensors first
        list_tokens=[]
        list_segments=[]
        list_labels=[]
        list_eye=[]

        max_seq_length=tokens.shape[1]
        # Create a batch for each token in sentence
        for idx,text in enumerate(tokens):

            # Check how many non-zero tokens are in this text (including special tokens)
            seq_length=sum((text != pad_token_id).int()).item()
            # We repeat the input text for each actual word in sentence
            # -2 because we do not run BERT for <CLS> and <SEP>
            inputs  = text.repeat(seq_length- 2, 1)

            # We do the same for the segments, which are just zero
            segments_tensor = torch.zeros(seq_length-2, max_seq_length, dtype=torch.int64)

            # Create the basis version of our labels
            labels_tensor = inputs.clone()

            # Create Masking matrix
            # First, how many tokens to mask?
            nr_mask_tokens = seq_length - 2
            # Create square eye matrix of max_sequence length
            # But cut both the first token, as well as passing tokens as rows
            eye_matrix = torch.eye(max_seq_length, dtype=torch.bool)[1:seq_length-1, :]

            # We Mask diagonal tokens
            inputs[eye_matrix] = mask_token_id
            # Set all other labels to -1
            labels_tensor[~eye_matrix] = -1

            # Append lists
            list_tokens.append(inputs)
            list_segments.append(segments_tensor)
            list_labels.append(labels_tensor)
            list_eye.append(eye_matrix.int())


        tokens = torch.tensor([], requires_grad=False)
        segments = torch.tensor([], requires_grad=False)
        labels = torch.tensor([], requires_grad=False)
        eyes = torch.tensor([], requires_grad=False)

        # Send to GPU
        tokens = torch.cat(list_tokens).to(device)
        segments=torch.cat(list_segments).to(device)
        labels=torch.cat(list_labels).to(device)
        eyes=torch.cat(list_eye).to(device)

        # Save some memory insallah
        del list_tokens
        del list_labels
        del list_segments
        del list_eye

        # Get predictions
        bert.eval()
        with torch.no_grad():
            loss, predictions, attn= bert(tokens, masked_lm_labels=labels, token_type_ids=segments)

        del tokens
        del labels
        del segments

        # Only return predictions of masked words (gives one per word for each sentence)
        predictions = predictions[eyes.bool(), :]

        if return_max==True:
            predictions=torch.argmax(predictions, dim=1)


        # Softmax prediction probabilities
        softmax=torch.nn.Softmax(dim=1)
        predictions=softmax(predictions)


        # Operate on attention
        attn=torch.stack(attn)
        # Max over layers and attention heads
        attn,_ =torch.max(attn,dim=0)
        attn,_ = torch.max(attn, dim=1)
        # We are left with a one matrix for each batch
        # Select the attention of the focal tokens only
        attn=attn[eyes.bool(),:]
        # Note attention is now of (nr_tokens,sequence_size+2)
        # because we added <SEP> and <CLS> tokens
        return predictions.cpu(), attn.cpu()

    # %% Utilities
    def calculate_cutoffs(self,x, method="mean", percent=100, max_degree=100,min_cut=0.001):
        """
        Different methods to calculate cutoff probability and number.

        :param x: Contextual vector
        :param method: mean: Only accept entries above the mean; percent: Take the k biggest elements that explain X% of mass.
        :return: cutoff_number and probability
        """
        if method == "mean":
            cutoff_probability = max(np.mean(x), min_cut)
            cutoff_number = max(np.int(len(x) / 100), 100)
        elif method == "percent":
            sortx = np.sort(x)[::-1]
            cum_sum = np.cumsum(sortx)
            cutoff = cum_sum[-1] * percent/100
            cutoff_number = np.where(cum_sum >= cutoff)[0][0]
            if cutoff_number == 0: cutoff_number=max_degree
            cutoff_probability = min_cut
        else:
            cutoff_probability = min_cut
            cutoff_number = 0

        return min(cutoff_number,max_degree), cutoff_probability

    
    def get_weighted_edgelist(self,token, x, time, cutoff_number=100, cutoff_probability=0, seq_id=0, pos=0,max_degree=100):
        """
        Sort probability distribution to get the most likely neighbor nodes.
        Return a networksx weighted edge list for a given focal token as node.

        :param token: Numerical, token which to add
        :param x: Probability distribution
        :param cutoff_number: Number of neighbor token to consider. Not used if 0.
        :param cutoff_probability: Lowest probability to consider. Not used if 0.
        :return: List of tuples compatible with networkx
        """
        # Get the most pertinent words
        if cutoff_number > 0:
            neighbors = np.argsort(-x)[:cutoff_number]
        else:
            neighbors = np.argsort(-x)[:max_degree]

        # Cutoff probability (zeros)
        # 20.08.2020 Added simple norm
        if len(neighbors > 0):
            if cutoff_probability>0:
                neighbors = neighbors[x[neighbors] > cutoff_probability]
            weights = simple_norm(x[neighbors])
            return [(int(token), int(x[0]), int(time), {'weight': float(x[1]), 'p1': int(seq_id), 'p2': int(pos)}) for x in list(zip(neighbors, weights))]
        else:
            return None
