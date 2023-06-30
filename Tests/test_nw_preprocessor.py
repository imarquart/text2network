import logging
from pathlib import Path

import pytest

from text2network.utils.file_helpers import check_create_folder
from text2network.preprocessing.nw_preprocessor import nw_preprocessor


@pytest.mark.usefixtures("test_config")
def test_preprocess(test_config):
    config = test_config
    text_folders = check_create_folder(config["Paths"]["import_folder"], False)
    preprocessor = nw_preprocessor(config)
    preprocessor.preprocess_folders(text_folders, overwrite=True)
    db_file = check_create_folder(config["Paths"]["database"], False)
    db_file = Path(db_file)
    assert db_file.is_file()


@pytest.mark.usefixtures("test_db")
def test_text(test_db):
    db_file = test_db
    with tables.open_file(db_file, mode="r") as table:
        data = table.root.textdata.table
        sentence1 = data.read_where('(year==2020) & (p1==b"vol1")')[0]["text"]
        assert (
            sentence1 == b"a good leader is a good communicator."
        ), logging.error(
            "Sentence: {} \n Expected instead: {}".format(
                sentence1, b"A good leader is a good communicator."
            )
        )
