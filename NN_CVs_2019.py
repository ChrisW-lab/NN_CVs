import tensorflow as tf
from tensorflow.keras import layers, optimizers
import nltk
import refine_data_set
import data_organise
import text_vectorise
import random

lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')


if __name__ == '__main__':
    raw_X, raw_Y = refine_data_set.refine_X_Y()

    balanced_X, balanced_Y = data_organise.balance_data_binary(raw_X, raw_Y)

    # count = 0
    # for i in range(10000):
    #     j = random.randint(0, len(balanced_X) - 1)
    #     if raw_Y[raw_X.index(balanced_X[j])] != balanced_Y[j]:
    #         print(raw_Y[raw_X.index(balanced_X[j])])
    #         print(balanced_Y[j])
    #         count += 1
    # print('Missed ', count, ' in 10000')

    # print('There are ', balanced_Y.count(0), '\'0s\' in balanced_Y')
    # print('There are ', balanced_Y.count(1), '\'1s\' in balanced_Y')

    X_vectorised, word_index_map = text_vectorise.vectorise(balanced_X, lemmatizer, stop_words, 2, twit=True)
    # split_train_test shuffles and splits

    X_train, Y_train, X_test, Y_test = data_organise.split_train_test(X_vectorised, balanced_Y, split=0.7)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape) + ' of type: ', type(X_train))
    print("Y_train shape: " + str(Y_train.shape) + ' of type: ', type(Y_train))
    print("X_test shape: " + str(X_test.shape) + ' of type: ', type(X_test))
    print("Y_test shape: " + str(Y_test.shape) + ' of type: ', type(Y_test))

    model = tf.keras.Sequential()
    model.add(layers.Dense(25, activation='relu', input_dim=X_vectorised.shape[1]))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_data=(X_test, Y_test))
