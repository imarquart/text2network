import json
import logging
import os
import pickle
import re
import unicodedata

from bs4 import BeautifulSoup
from nltk import sent_tokenize
from tqdm import tqdm

from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import log, setup_logger

logger = logging.getLogger("t2n")


class TextPreprocessor:
    def __init__(
        self,
        maximum_sequence_length,
        split_symbol,
        logging_level=logging.INFO,
        other_loggers=logging.WARNING,
        input_folder=None,
        output_folder=None,
        max_json_length=1000000,
    ):
        self.maximum_sequence_length = maximum_sequence_length
        self.split_symbol = split_symbol
        self.logging_level = logging_level
        self.other_loggers = other_loggers
        self.folder = input_folder
        self.output_folder = output_folder
        self.max_json_length = max_json_length

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
            json_length_counter = 0
            json_file_counter = 1

            logger.debug(f"Processing year {year}")

            for file in files:
                logger.debug(f"Processing file {file}")
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
                                "parameters": file.split(".txt")[0].split(self.split_symbol),
                                "sentence": split_sentence,
                                "index": total_index,
                                "year_index": year_index,
                                "file_index": file_index,
                            }
                            processed_sentences.append(metadata)
                            total_index += 1
                            year_index += 1
                            file_index += 1
                            json_length_counter += 1

                            # Save the sentences in batches
                            if json_length_counter >= self.max_json_length:
                                output_file = os.path.join(output_folder, f"{year}/")
                                output_file = check_create_folder(output_file, create_folder=True)
                                output_file = os.path.join(output_file, f"{json_file_counter}.json")
                                logger.debug(
                                    f"Json length: {json_length_counter} reached, saving file to disk in {output_file}"
                                )
                                with open(output_file, "w", encoding="utf-8") as f:
                                    json.dump(
                                        processed_sentences,
                                        f,
                                        ensure_ascii=False,
                                        indent=4,
                                    )
                                processed_sentences = []
                                json_length_counter = 0
                                json_file_counter += 1

            # Save the last sentences
            if len(processed_sentences) > 0:
                output_file = os.path.join(output_folder, f"{year}/")
                output_file = check_create_folder(output_file, create_folder=True)
                output_file = os.path.join(output_file, f"{json_file_counter}.json")
                logger.debug(
                    f"Sentences remaining: {len(processed_sentences)}. Saving to disk in {output_file}"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(processed_sentences, f, ensure_ascii=False, indent=4)

            # Create Metadata
            metadata = {"year": year, "len_year": year_index, "json_files": json_file_counter}
            # Pickle metadata
            if year_index > 0:
                output_file = os.path.join(output_folder, f"{year}/metadata.pkl")
                output_file = check_create_folder(output_file, create_folder=True)
                with open(output_file, "wb") as f:
                    pickle.dump(metadata, f)
