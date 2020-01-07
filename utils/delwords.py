import numpy as np
from string import punctuation

def create_stopword_list(tokenizer):
    """
    Return a list of tokenized tokens for words which should be removed from the data set

    :param tokenizer: BERT tokenizer
    :return: Numerical list of tokens to take out of sample
    """
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    alphabet_ids = tokenizer.convert_tokens_to_ids(alphabet)
    numbers = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'one', 'two', 'three', 'four']
    numbers.extend(list(punctuation))
    numbers_ids = tokenizer.convert_tokens_to_ids(numbers)
    pronouns = ['we', 'us', 'my', 'yourself', 'you', 'me', 'he', 'her', 'his', 'him', 'she', 'they', 'their', 'them',
                'me', 'myself', 'himself', 'herself', 'themselves']

    numericals=list(range(0,3000))
    numericals=[str(x) for x in numericals]
    numericals_ids = tokenizer.convert_tokens_to_ids(numericals)

    pronouns_ids = tokenizer.convert_tokens_to_ids(pronouns)
    stopwords = ['¨','@','110','100','1000','000','!','ve','&', ',', '.', 'and', '-', 'the', '##d', '...', 'that', 'to', 'as', 'for', '"', 'in', "'", 'a', 'of',
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
                 'than', 't', 'surely', '|', '#', 'till', '##ly', '_', 'al','«','»','{','[',']','}','%','+','-','>','<',':','.','=']
    stopwords_ids = tokenizer.convert_tokens_to_ids(stopwords)

    # Now all special tokens
    hash_ids=[v for k, v in tokenizer.vocab.items() if '##' in k]


    delwords = np.union1d(stopwords_ids, pronouns_ids)
    delwords = np.union1d(delwords, hash_ids)
    delwords = np.union1d(delwords, numbers_ids)
    delwords = np.union1d(delwords, alphabet_ids)
    delwords = np.union1d(delwords, numbers_ids)
    return delwords