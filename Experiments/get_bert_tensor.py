import torch

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
    with torch.no_grad():
        loss, predictions, attn= bert(tokens, masked_lm_labels=labels, token_type_ids=segments)

    del tokens
    del labels
    del segments

    # Only return predictions of masked words (gives one per word for each sentence)
    predictions = predictions[eyes.bool(), :]

    if return_max==True:
        predictions=torch.argmax(predictions, dim=1)

    # Operate on attention
    attn=torch.stack(attn)
    # Max over layers and attention heads
    attn,_ =torch.max(attn,dim=0)
    attn,_ = torch.max(attn, dim=1)
    # We are left with a one matrix for each batch size
    attn=attn[eyes.bool(),:]
    return predictions.to(torch.device('cpu')), attn.cpu()