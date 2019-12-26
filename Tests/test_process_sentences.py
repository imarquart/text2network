import os
import time
from NLP.src.text_processing.process_sentences import process_sentences
from NLP.utils.load_bert import get_bert_and_tokenizer_local
import tables
# os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')
import subprocess

MAX_SEQ_LENGTH = 30
batch_size = 36
method = "context_element"
cwd = os.getcwd()
text_db = os.path.join(cwd, 'NLP/data/texts.h5')
modelpath = os.path.join(cwd, 'NLP/models')

tensor_path = ''.join(['NLP/data'])
tensor_path = os.path.join(cwd, tensor_path)

#tensor_db = ''.join(['NLP/data/tensor_db_', method, '.h5'])
#tensor_db = os.path.join(cwd, tensor_db)

tensor_db = ''.join(['E:/NLP/tensor_db_', method, '.h5'])
temp_db = ''.join(['E:/NLP/temp_db_', method, '.h5'])

filters = tables.Filters(complevel=9, complib='blosc')
ch_shape = (100,)

if __name__ == '__main__':
    start_time = time.time()

    tokenizer, bert = get_bert_and_tokenizer_local(modelpath)
    DICT_SIZE = tokenizer.vocab_size

    process_sentences(tokenizer, bert, text_db, tensor_db,temp_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size, nr_workers=0,
                      copysort=True, method=method, filters=filters, ch_shape=ch_shape)

    data_file = tables.open_file(tensor_db, mode="r", title="Data File")
    print(data_file)
    print(data_file.root.token_data.table)

print("Total Time: %s seconds" % (time.time() - start_time))
