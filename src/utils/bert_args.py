
class bert_args():
    def __init__(self,database,where_string, output_dir, pretrained_dir, do_train= True, mlm_probability=0.15, block_size=30,
                 loss_limit=0.5, gpu_batch=4, epochs=1, warmup_steps=0, save_steps=50000, eval_steps=50,eval_loss_limit=0.5,logging_level=40):
        """
        Normally, PyTorch Transformers loads parameters via the commandline.
        This class serves the same purpose
        :param train_data_file:
        :param output_dir:
        :param do_train:
        :param model_dir:
        :param mlm_probability:
        :param block_size:
        :param loss_limit:
        :param gpu_batch:
        :param epochs:
        :param warmup_steps:
        :param save_steps:
        """
        #self.train_data_file = train_data_file
        #self.eval_data_file = train_data_file
        self.output_dir = output_dir

        self.database=database
        self.where_string=where_string


        self.mlm = True
        self.mlm_probability = mlm_probability

        self.loss_limit = loss_limit
        self.logging_level=logging_level
        if do_train == True:
            self.do_train = True
            self.do_eval = True
        else:
            self.do_train = False
            self.do_eval = True

        self.do_lower_case = True

        self.model_name_or_path = pretrained_dir
        self.model_dir = pretrained_dir

        self.warmup_steps = warmup_steps
        self.num_train_epochs = epochs

        self.block_size = block_size

        self.evaluate_during_training = True

        self.per_gpu_train_batch_size = gpu_batch
        self.per_gpu_eval_batch_size = gpu_batch

        self.model_type = "bert"
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_steps = -1
        self.logging_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = "01"
        self.local_rank = -1
        self.n_gpu = 1
        self.server_ip = ''
        self.eval_loss_limit=eval_loss_limit
        self.server_port = ''

        self.device = None
