import nltk
import numpy as np


def tokenizer(text, lemmatizer, stop_words, min_length, twitter=True):
    '''Tokenizes a string of text.  Removes Twitter handles if twitter = True'''
    # site Lazy Programmer although made some changes
    text = text.lower()
    tokens = nltk.tokenize.word_tokenize(text)
    if twitter:
        tokens = [t for t in tokens if '@' not in t]
    tokens = (t for t in tokens if len(t) > min_length)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if tokens == []:
        tokens = ['']
    return tokens


def vectorise(X, lemmatizer, stop_words, min_length, twit=True):
    '''Takes a list of strings (X) and returns a list of vectors representing the text'''
    word_index_map = {}
    current_index = 0
    X_tokenized = []

    for string in X:
        tokens = tokenizer(string, lemmatizer, stop_words, min_length, twitter=twit)
        X_tokenized.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1

    data = np.zeros((len(X), len(word_index_map)))
    row = 0

    for tokens in X_tokenized:
        x = np.zeros(len(word_index_map))
        for token in tokens:
            i = word_index_map[token]
            x[i] += 1
        x = x / x.sum()  # normalising each vector to add up to 1
        data[row, :] = x
        row += 1

    return data, word_index_map
