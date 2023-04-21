import logging
import os
import json
import re
from tqdm import tqdm
from nltk import sent_tokenize
from bs4 import BeautifulSoup
import unicodedata

from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import log, setup_logger


class TextPreprocessor:
    def __init__(
        self, maximum_sequence_length, split_symbol, folder=None, output_folder=None
    ):
        self.maximum_sequence_length = maximum_sequence_length
        self.split_symbol = split_symbol
        self.logging_level = logging.INFO
        self.other_loggers = logging.WARNING
        self.folder = folder
        self.output_folder = output_folder

    def _split_sentence(self, sentence):
        if len(sentence) <= self.maximum_sequence_length:
            return [sentence]

        half = len(sentence) // 2
        sentence_1 = sentence[:half].strip()
        sentence_2 = sentence[half:].strip()

        return self._split_sentence(sentence_1) + self._split_sentence(sentence_2)

    def _check_and_fix_encoding(self, text):
        return unicodedata.normalize("NFKD", text)

    def _replace_line_breaks_and_whitespace(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _remove_unnecessary_characters(self, text):
        return re.sub(r"[#(){}]", "", text)

    def _remove_html_fragments(self, text):
        return BeautifulSoup(text, "html.parser").get_text()

    def _remove_numerics(self, text):
        text = re.sub(r"^\d+", "", text)
        text = re.sub(r"\d+$", "", text)
        text = re.sub(r"\b\d{5,}\b", "", text)
        return text

    def _preprocess(self, text):
        text = self._check_and_fix_encoding(text)
        text = self._replace_line_breaks_and_whitespace(text)
        text = self._remove_unnecessary_characters(text)
        text = self._remove_html_fragments(text)
        text = self._remove_numerics(text)
        return text

    @log()
    def preprocess(self, folder=None, output_folder=None):
        if not folder:
            folder = self.folder
        if not output_folder:
            output_folder = self.output_folder
        assert folder, "No folder specified"
        assert output_folder, "No output folder specified"

        total_index = 0
        for subdir, dirs, files in tqdm(os.walk(folder)):
            year = os.path.basename(subdir)
            processed_sentences = []
            year_index = 0

            for file in files:
                file_index = 0
                if file.endswith(".txt"):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    processed_content = self._preprocess(content)
                    sentences = sent_tokenize(processed_content)

                    for sentence in sentences:
                        split_sentences = self._split_sentence(sentence)
                        for split_sentence in split_sentences:
                            metadata = {
                                "filename": file,
                                "year": year,
                                "parameters": file.split(".txt")[0].split(
                                    self.split_symbol
                                ),
                                "sentence": split_sentence,
                                "index": total_index,
                                "year_index": year_index,
                                "file_index": file_index,
                            }
                            processed_sentences.append(metadata)
                            total_index += 1
                            year_index += 1
                            file_index += 1

                output_folder = check_create_folder(output_folder, create_folder=True)
                output_file = os.path.join(output_folder, f"{year}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(processed_sentences, f, ensure_ascii=False, indent=4)
