import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from keras.layers import Embedding, LSTM, Bidirectional, GRU

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from keras.models import Model
import matplotlib
matplotlib.use('TkAgg')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# Set random seed (for reproducibility)
np.random.seed(1000)
from sklearn.preprocessing import LabelEncoder
import os

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


def label_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    return integer_encoded

def one_hot_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    ohe_features = np.eye(8)[integer_encoded]
    return ohe_features


use_gpu = True
config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)


corpusList = []
labelList = []

list_full_date = []

l_train_df = pd.read_csv(filePath + "/train/sentiments/train-dataset/LokShobaElc2019Cong-moods-train.csv").dropna()
l_test_df =  pd.read_csv(filePath + "/train/sentiments/test-dataset/LokShobaElc2019Cong-moods-test.csv").dropna()


list_full_date.append(l_train_df)
list_full_date.append(l_test_df)

combined_df = pd.concat(list_full_date, axis = 0, ignore_index = True)

# Keras LSTM model
batch_size = 64
nb_epochs = 5
NO_CLASSES = 8

# Use the Keras tokenizer
num_words = 20000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(combined_df['tweet'].values)
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))


# Pad the data
X = tokenizer.texts_to_sequences(combined_df['tweet'].values)
X = pad_sequences(X, maxlen=2000)
Y = pd.get_dummies(combined_df['mood']).values


# Build out our simple LSTM
embed_dim = 128
lstm_out = 256

avg_length = 0.0
max_length = 0

corpus = combined_df['tweet'].values
print('Corpus size: {}'.format(len(corpus)))

# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)

for tweet in tokenized_corpus:
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))

print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
print('Max tweet length: {}'.format(max_length))


def lstm_gru_model():

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length=X.shape[1]))
    model.add(Bidirectional(GRU(lstm_out,  recurrent_dropout=0.2, dropout=0.2, activation='tanh')))

    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])
    print(model.summary())


    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    output_test = model.predict(X_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in Y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify W2Vec LSTM Precision =", precisions)
    print("Classify W2Vec LSTM Recall=", recall)
    print("Classify W2Vec LSTM F1 Score =", f1_score)


    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0,8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))

def bi_lstm_model():

    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length=X.shape[1]))
    model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2)))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    output_test = model.predict(X_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r == 1)[0][0] for r in Y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify W2Vec Bi-LSTM Precision =", precisions)
    print("Classify W2Vec Bi-LSTM Recall=", recall)
    print("Classify W2Vec Bi-LSTM F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))

def lstm_model():

    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(8,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
    print(model.summary())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    output_test = model.predict(X_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in Y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify W2Vec LSTM Precision =", precisions)
    print("Classify W2Vec LSTM Recall=", recall)
    print("Classify W2Vec LSTM F1 Score =", f1_score)


    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0,8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))


if __name__ == '__main__':
    lstm_gru_model()
    lstm_model()
    bi_lstm_model()

