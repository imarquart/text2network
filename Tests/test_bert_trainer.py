import pytest

from text2network.utils.file_helpers import check_create_folder


@pytest.mark.usefixtures("get_bert_trainer")
def test_split_hierarchy(get_bert_trainer):
    trainer=get_bert_trainer
    uniques=trainer.get_uniques(trainer.split_hierarchy)
    print(uniques)
    assert len(uniques["file"])==2


@pytest.mark.usefixtures("get_bert_trainer")
def test_consistent_vocab(get_bert_trainer):
    trainer=get_bert_trainer
    uniques = trainer.get_uniques(trainer.split_hierarchy)
    queries = [x[0] for x in uniques["query_filename"]]
    tokenizer, missing_tokens, total_tokens = trainer.get_consistent_vocabulary(queries)
    assert 'spitzentransformer' in missing_tokens, "Missing word not found when checking tokenizer consistency"
    assert 'fehlworttest' in missing_tokens, "Missing word not found when checking tokenizer consistency"
    assert 'spitzentransformer' in tokenizer.vocab, "Missing word not in vocab, even if found"
    assert 'fehlworttest' in tokenizer.vocab, "Missing word not in vocab, even if found"


@pytest.mark.usefixtures("get_bert_trainer","get_training_tokenizer_folder")
def test_training_and_prediction(get_bert_trainer, get_training_tokenizer_folder):
    trainer=get_bert_trainer
    tokenizer_folder=get_training_tokenizer_folder
    uniques = trainer.get_uniques(trainer.split_hierarchy)
    queryf = list(uniques["query_filename"])[0]
    print(queryf)
    query = queryf[0]
    fname = queryf[1]
    bert_folder = ''.join([trainer.trained_folder, '/', fname])
    bert_folder = check_create_folder(bert_folder, create_folder=True)
    results = trainer.train_one_bert(query,bert_folder,tokenizer_folder)
    print(results)
    assert results['eval_loss_'] <= trainer.bert_config.getfloat(
                             'loss_limit')

