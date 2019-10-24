import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def get_bert_tensor(tokenizer, bert,texts,MAX_SEQ_LENGTH,return_max=False):
    # DONE: Filter out non-needed rows where no masked token exist
    # TODO: Check punctuation
    # TODO: CPU / TPU pushing
    # DONE: Return token_id list

    # We use lists of tensors first because we want to pad sequences, at the same time add dimensions
    list_tokens=[]
    list_segments=[]
    list_labels=[]
    list_eye=[]
    token_id=[]
    # Our labels for the batches will stay lists
    batch_label=[]
    batch_size=[]

    # First we tokenize each sentence
    for text in texts:
        indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
        inputs = torch.tensor([indexed_tokens], requires_grad=False)
        inputs =inputs.squeeze(dim=0).squeeze()
        list_tokens.append(inputs)
        # Also save list of tokens
        token_id.append(inputs[1:-1])

    # Use pad_sequence to stack tokens
    tokens = pad_sequence(list_tokens, batch_first=True,padding_value=tokenizer.pad_token_id)
    # How long are the resulting sequences?
    seq_length=tokens.shape[1]

    # Unload token list
    list_tokens=[]


    # Create a batch for each token in sentence
    for idx,text in enumerate(tokens):

        # We repeat the input text for each actual word in sentence
        # -2 because we do not run BERT for <CLS> and <SEP>
        inputs  = text.repeat(seq_length- 2, 1)

        # We do the same for the segments, which are just zero
        segments_tensor = torch.zeros(seq_length, dtype=torch.int64)
        segments_tensor = segments_tensor.repeat(seq_length - 2, 1)

        # Create the basis version of our labels
        labels_tensor = inputs.clone()

        # Create Masking matrix
        # First, how many tokens to mask initially?
        nr_mask_tokens = seq_length - 2
        # Create eye matrix
        eye_matrix = torch.eye(seq_length, dtype=torch.bool)[1:-1, :]
        # Take out padding tokens
        eye_matrix[inputs == 0] = 0
        # Take out <SEP> tokens
        eye_matrix[inputs == tokenizer.sep_token_id] = 0
        # Recalculate correct number of masking steps
        nr_mask_tokens=eye_matrix.int().sum().item()
        # We Mask diagonal tokens
        inputs[eye_matrix] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        # Set all other labels to -1
        labels_tensor[~eye_matrix] = -1

        # Save batch id's and labels
        batch_idx=np.repeat(idx, nr_mask_tokens)
        batch_label=np.append(batch_label,batch_idx)
        batch_size=np.int64(np.append(batch_size,nr_mask_tokens))

        # Append lists
        list_tokens.append(inputs)
        list_segments.append(segments_tensor)
        list_labels.append(labels_tensor)
        list_eye.append(eye_matrix)

    batch_label=np.int64(batch_label)

    tokens = torch.tensor([], requires_grad=False)
    segments = torch.tensor([], requires_grad=False)
    labels = torch.tensor([], requires_grad=False)
    eyes = torch.tensor([], requires_grad=False)

    tokens = torch.cat(list_tokens)
    segments=torch.cat(list_segments)
    labels=torch.cat(list_labels)
    eyes=torch.cat(list_eye)

    # Save some memory insallah
    del list_tokens
    del list_labels
    del list_segments
    del list_eye


    with torch.no_grad():
        loss, predictions = bert(tokens, masked_lm_labels=labels, token_type_ids=segments)

    # Apply softmax to graph the distributions
    # WILL DO THIS LATER BC OF LINEARITY
    #softmax = torch.nn.Softmax(dim=2)
    #predictions = softmax(predictions)

    # Only return predictions of masked words (gives one per word for each sentence)
    predictions = predictions[eyes, :]

    if return_max==True:
        predictions=torch.argmax(predictions, dim=1)

    return token_id,batch_size,batch_label,predictions