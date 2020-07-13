from refine_data_set import refine_X_Y
import text_vectorise
import data_organise
import nltk
import numpy as np
from tf_model_1 import tf_model_1

lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')

if __name__ == '__main__':
    raw_X, raw_Y = refine_X_Y()
    # raw_X, raw_Y both lists with len 2273.  Y has 361 positive classes.  optional balance below

    # X, classes = data_organise.balance_data(raw_X, raw_Y)
    Y = data_organise.convert_to_one_hot(raw_Y, C=2)
    print('Y is of shape: ', np.shape(Y))
    X_vectorised, word_index_map = text_vectorise.vectorise(raw_X, lemmatizer, stop_words, 2, twit=False)
    print('X_vectorised is of shape: ', np.shape(X_vectorised))
    print(word_index_map)
    with open('./word_index_map.txt', 'w', encoding='utf-8') as stream:
        stream.write(str(word_index_map))
    # print('word_index_map saved')

    X_train, Y_train, X_test, Y_test = data_organise.split_train_test(X_vectorised, Y, split=0.7)

    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape) + ' of type: ', type(X_train))
    print("Y_train shape: " + str(Y_train.shape) + ' of type: ', type(Y_train))
    print("X_test shape: " + str(X_test.shape) + ' of type: ', type(X_test))
    print("Y_test shape: " + str(Y_test.shape) + ' of type: ', type(Y_test))

    parameters = tf_model_1(X_train, Y_train, X_test, Y_test)
