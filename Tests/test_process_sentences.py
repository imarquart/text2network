
import os
import time
from NLP.Experiments.process_sentences import process_sentences
from NLP.utils.load_bert import get_bert_and_tokenizer
os.chdir('/home/ingo/PhD/BERT-NLP/NLP')
start_time = time.time()
cwd= os.getcwd()
text_db=os.path.join(cwd,'data/texts.h5')
tensor_db=os.path.join(cwd,'data/tensor_db.h5')
modelpath=os.path.join(cwd,'models')
MAX_SEQ_LENGTH=50
batch_size=10
tokenizer, bert = get_bert_and_tokenizer(modelpath)
DICT_SIZE=tokenizer.vocab_size
batch=process_sentences(tokenizer, bert, text_db, tensor_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size)


print("--- %s seconds ---" % (time.time() - start_time))