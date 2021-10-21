from string import punctuation

import numpy as np


def list_stopwords():
    nltk_stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
                      "aren",
                      "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                      "but", "by", "can",
                      "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing",
                      "don", "don't",
                      "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
                      "hasn't",
                      "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
                      "his", "how", "i",
                      "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma",
                      "me", "mightn",
                      "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
                      "not", "now", "o",
                      "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
                      "own", "re", "s",
                      "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so",
                      "some", "such", "t",
                      "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
                      "these", "they",
                      "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
                      "wasn't", "we",
                      "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
                      "will", "with", "won",
                      "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                      "yourself",
                      "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've",
                      "let's", "ought",
                      "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd",
                      "we'll", "we're",
                      "we've", "what's", "when's", "where's", "who's", "why's", "would"]
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'one', 'two', 'three', 'four']
    numbers.extend(list(punctuation))
    pronouns = ['we', 'us', 'my', 'yourself', 'you', 'me', 'he', 'her', 'his', 'him', 'she', 'they', 'their', 'them',
                'me', 'myself', 'himself', 'herself', 'themselves', 'your', 'mine']
    numericals = list(range(0, 3000))
    numericals = [str(x) for x in numericals]
    stopwords = ['”', 'could', 'should', '¨', '@', '110', '100', '1000', '000', '!', 've', '&', ',', '.', 'and', '-',
                 'the',
                 '##d', '...', 'that', 'to', 'as', 'for', '"', 'in', "'", 'a', 'of',
                 'only', ':', 'so', 'all', 'one', 'it', 'then', 'also', 'with', 'but', 'by', 'on', 'just', 'like',
                 'again', ';', 'more', 'this', 'not', 'is', 'there', 'was', 'even', 'still', 'after', 'here', 'later',
                 '!', 'over', 'from', 'i', 'or', '?', 'at', 'first', '##s', 'while', ')', 'before', 'when', 'once',
                 'too', 'out', 'yet', 'because', 'some', 'though', 'had', 'instead', 'always', '(', 'well', 'back',
                 'tonight', 'since', 'about', 'through', 'will', 'them', 'left', 'often', 'what', 'ever', 'until',
                 'sometimes', 'if', 'however', 'finally', 'another', 'somehow', 'everything', 'further', 'really',
                 'last', 'an', '/', 'rather', 's', 'may', 'be', 'each', 'thus', 'almost', 'where', 'anyway', 'their',
                 'has', 'something', 'already', 'within', 'any', 'indeed', '##a', '[UNK]', '~', 'every', 'meanwhile',
                 'would', '##e', 'have', 'nevertheless', 'which', 'how', '1', 'are', 'either', 'along', 'thereafter',
                 'otherwise', 'did', 'quite', 'these', 'can', '2', 'its', 'merely', 'actually', 'certainly', '3',
                 'else', 'upon', 'except', 'those', 'especially', 'therefore', 'beside', 'apparently', 'besides',
                 'third', 'whilst', '*', 'although', 'were', 'likewise', 'mainly', 'four', 'seven', 'into', 'm', ']',
                 'than', 't', 'surely', '|', '#', 'till', '##ly', '_', 'al', '«', '»', '{', '[', ']', '}', '%', '+',
                 '-', '>', '<', ':', '.', '=']

    delwords = np.union1d(stopwords, pronouns)
    delwords = np.union1d(delwords, nltk_stopwords)
    delwords = np.union1d(delwords, numbers)
    delwords = np.union1d(delwords, alphabet)
    delwords = np.union1d(delwords, numericals)
    return delwords


def create_stopword_list(tokenizer):
    """
    Return a list of tokenized tokens for words which should be removed from the data set

    :param tokenizer: BERT tokenizer
    :return: Numerical list of tokens to take out of sample
    """
    delwords = list_stopwords()
    delwords_ids = tokenizer.convert_tokens_to_ids(delwords)
    return delwords_ids


def create_stopword_strings():
    """
    Return a list of tokenized tokens for words which should be removed from the data set

    :param tokenizer: BERT tokenizer
    :return: Numerical list of tokens to take out of sample
    """
    return list_stopwords()
