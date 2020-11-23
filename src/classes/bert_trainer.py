from src.utils.hash_file import hash_file, check_step, complete_step
from src.utils.load_bert import get_bert_and_tokenizer
from src.utils.bert_args import bert_args
from src.datasets.text_dataset import bert_dataset
from src.classes.run_bert import run_bert
import torch

class bert_trainer():
    def __init__(self, db_folder, pretrained_folder, trained_folder, bert_config):

        # Load Tokenizer
        tokenizer, model = get_bert_and_tokenizer(pretrained_folder, True)

        where_string='(p1==b"1") & (p2==b"Washington")'
        dataset=bert_dataset(tokenizer,db_folder,where_string,30)
        bert_folder=''.join([trained_folder,'/test'])

        args = bert_args(db_folder,  where_string,bert_folder, pretrained_folder, mlm_probability=bert_config.getfloat('mlm_probability'),
                         block_size=bert_config.getint('max_seq_length'),
                         loss_limit=bert_config.getfloat('loss_limit'), gpu_batch=bert_config.getint('gpu_batch'),  epochs=bert_config.getint('epochs'),
                         warmup_steps=bert_config.getint('warmup_steps'), save_steps=bert_config.getint('save_steps'))

        torch.cuda.empty_cache()

        results = run_bert(args)

        print("BERT results %s" % results)
