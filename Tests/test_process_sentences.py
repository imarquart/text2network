
import os
import time
from NLP.Experiments.process_sentences import process_sentences
from NLP.utils.load_bert import get_bert_and_tokenizer_local
#os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')
if __name__ == '__main__':
    start_time = time.time()
    cwd= os.getcwd()
    text_db=os.path.join(cwd,'NLP/data/texts.h5')
    tensor_db=os.path.join(cwd,'NLP/data/tensor_db.h5')
    modelpath=os.path.join(cwd,'NLP/models')
    MAX_SEQ_LENGTH=30
    batch_size=3
    tokenizer, bert = get_bert_and_tokenizer_local(modelpath)
    DICT_SIZE=tokenizer.vocab_size

    process_sentences(tokenizer, bert, text_db, tensor_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size)


    print("--- %s seconds ---" % (time.time() - start_time))