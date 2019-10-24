import torch
import numpy as np
import tables

from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import BertConfig

filepath = 'data_file.h5'
# Load models
output_vocab_file='D:/NLP/BERT-NLP/BERT-NLP/models/bert-base-uncased-vocab.txt'
output_model_file = "D:/NLP/BERT-NLP/BERT-NLP/models/bert-base-uncased-pytorch_model.bin"
output_config_file = "D:/NLP/BERT-NLP/BERT-NLP/models/bert-base-uncased-config.json"

tokenizer = BertTokenizer.from_pretrained(output_vocab_file)
config = BertConfig.from_json_file(output_config_file)
bert = BertForMaskedLM.from_pretrained(output_model_file,config=config)

# With online access
#tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','bert-base-uncased')  # Download vocabulary from S3 and cache.
#bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')


MAX_SEQ_LENGTH = 40
DICT_SIZE = tokenizer.vocab_size

# Enter sentence without punctuation (uncomment)
text_1 = "Barack Obama is the president of the United States."
text_2 = "Angela Merkel is still chancellor of Germany."
texts = [text_1, text_2]


class Seq_Particle(tables.IsDescription):
    seq_id = tables.UInt32Col()
    token_ids = tables.UInt32Col(shape=[1, MAX_SEQ_LENGTH])
    token_dist = tables.Float32Col(shape=(MAX_SEQ_LENGTH, DICT_SIZE))
    seq_size=tables.UInt32Col()


class Token_Seq_Particle(tables.IsDescription):
    token_id = tables.UInt32Col()
    seq_id = tables.UInt32Col()
    seq_size = tables.UInt32Col()
    pos_id = tables.UInt32Col()
    token_ids = tables.UInt32Col(shape=[1, MAX_SEQ_LENGTH])
    token_dist = tables.Float32Col(shape=(MAX_SEQ_LENGTH, DICT_SIZE))


class Token_Particle(tables.IsDescription):
    token_id = tables.UInt32Col()
    seq_id = tables.UInt32Col()
    pos_id = tables.UInt32Col()
    seq_size = tables.UInt32Col()
    own_dist = tables.Float32Col(shape=(1, DICT_SIZE))
    context_dist = tables.Float32Col(shape=(1, DICT_SIZE))


def process_sentences(tokenizer, bert, texts, seq_ids, filepath, batch_size):

    try:
        data_file = tables.open_file(filepath, mode="a", title="Data File")
    except:
        data_file = tables.open_file(filepath, mode="w", title="Data File")

    try:
        seq_table = data_file.root.seq_data.table
    except:
        group = data_file.create_group("/", 'seq_data', 'Sequence Data')
        seq_table = data_file.create_table(group, 'table', Seq_Particle, "Sequence Table")

    try:
        token_seq_table = data_file.root.token_seq_data.table
    except:
        group = data_file.create_group("/", 'token_seq_data', 'Token Sequence Data')
        token_seq_table = data_file.create_table(group, 'table', Token_Seq_Particle, "Token Sequence Table")

    try:
        token_table = data_file.root.token_data.table
    except:
        group = data_file.create_group("/", 'token_data', 'Token Data')
        token_table = data_file.create_table(group, 'table', Token_Particle, "Token Table")

    # TODO: Batching
    # TODO: Parallelize
    # for batch etc

    token_id, batch_size,batch_label,bert_tensor = get_bert_tensor(tokenizer, bert, texts, MAX_SEQ_LENGTH)



    for label_id in np.unique(batch_label):

        #%% Sequence Table
        # Init new row pointer
        particle = seq_table.row
        # Fill general information
        particle['seq_id'] = label_id
        particle['seq_size'] = batch_size[label_id]

        # Pad and add sequence of IDs
        idx=torch.zeros([1,MAX_SEQ_LENGTH],requires_grad=False,dtype=torch.int32)
        idx[0,:batch_size[label_id]]=token_id[label_id]
        particle['token_ids'] = idx
        # Pad and add distributions per token
        dists=torch.zeros([MAX_SEQ_LENGTH,DICT_SIZE],requires_grad=False)
        dists[:batch_size[label_id],:]=bert_tensor[batch_label == label_id, :]
        particle['token_dist'] = dists
        # Append
        particle.append()

        #%% Token-Sequence Table
        for pos,token in enumerate(token_id[label_id]):
            particle=token_seq_table.row
            particle['token_id'] = token
            particle['seq_id'] = label_id
            particle['seq_size'] = batch_size[label_id]
            particle['pos_id'] = pos
            particle['token_ids'] = idx
            particle['token_dist'] = dists
            particle.append()

        #%% Token Table
        context_index = np.zeros([MAX_SEQ_LENGTH], dtype=np.bool)
        context_index[:batch_size[label_id]] = True
        for pos,token in enumerate(token_id[label_id]):
            particle=token_table.row
            particle['token_id'] = token
            particle['seq_id'] = label_id
            particle['seq_size'] = batch_size[label_id]
            particle['own_dist'] = dists[pos,:].unsqueeze(0)
            context_index=np.arange(batch_size[label_id])!=pos
            context_index=np.concatenate([context_index,np.zeros([MAX_SEQ_LENGTH-batch_size[label_id]],dtype=np.bool)])
            context_dist=dists[context_index,:]
            particle['context_dist']=(torch.sum(context_dist,dim=0).unsqueeze(0))/batch_size[label_id]
            particle.append()

    data_file.flush()
    data_file.close()

    #h5file = tables.open_file("tutorial1.h5", mode="r", title="Test file")

    #table = h5file.root.detector.readout
    #test = [x['own_dist'] for x in table.iterrows()]
