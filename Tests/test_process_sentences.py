
import os
import time
from NLP.src.text_processing.process_sentences2 import process_sentences
from NLP.utils.load_bert import get_bert_and_tokenizer_local
#os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')

MAX_SEQ_LENGTH = 20
batch_size = 36
method = "attention"
cwd = os.getcwd()
text_db = os.path.join(cwd, 'NLP/data/texts.h5')
modelpath = os.path.join(cwd, 'NLP/models')
tensor_db = ''.join(['NLP/data/tensor_db_', method, '.h5'])
tensor_db = os.path.join(cwd, tensor_db)

if __name__ == '__main__':
    start_time = time.time()


    tokenizer, bert = get_bert_and_tokenizer_local(modelpath)
    DICT_SIZE=tokenizer.vocab_size

    process_sentences(tokenizer, bert, text_db, tensor_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size, nr_workers=0,copysort=True,method=method)


    print("Total Time: %s seconds" % (time.time() - start_time))