import numpy as np
import pandas as pd
import os
from nltk import tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input
from keras.layers import Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss, accuracy_score
from keras import initializers
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import layers

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


MAX_SENT_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000
NO_CLASSES = 8

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]

l_train_df = pd.read_csv(filePath + "/train/sentiments/train-dataset/LokShobaElc2019Cong-moods-train.csv").dropna()
l_test_df =  pd.read_csv(filePath + "/train/sentiments/test-dataset/LokShobaElc2019Cong-moods-test.csv").dropna()

labelList = []

list_train = []

list_train.append(l_train_df)
list_train.append(l_test_df)


combined_df = pd.concat(list_train, axis = 0, ignore_index = True)

def label_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    return integer_encoded


def one_hot_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    ohe_features = np.eye(NO_CLASSES)[integer_encoded]
    return ohe_features

train_mood = one_hot_encode(combined_df['mood'])
corpus = combined_df['tweet'].values
all_labels = train_mood

#Refr : https://github.com/richliao/textClassifier/issues/28
class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


tweets = []
labels = []
texts = []
# Compute average and max tweet length
avg_length = 0.0
max_length = 0

for idx in range(combined_df.tweet.shape[0]):
    tweet = combined_df.tweet[idx]
    texts.append(tweet)
    sentences = tokenize.sent_tokenize(tweet)
    tweets.append(sentences)

    labels.append(combined_df.mood[idx])
    if len(tweet) > max_length:
        max_length = len(tweet)
    avg_length += float(len(tweet))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = all_labels
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print (y_train.sum(axis=0))
print (y_test.sum(axis=0))

GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# Bidirectional LSTM
def biLSTM():

    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(128))(embedded_sequences)
    preds = Dense(NO_CLASSES, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print("model fitting - Bidirectional LSTM")
    model.summary()

    model.fit(x_train, y_train,
              nb_epoch=15, batch_size=64)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-LSTM Precision =", precisions)
    print("Classify Glove Bi-LSTM Recall=", recall)
    print("Classify Glove Bi-LSTM F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


def biGRUAttlayer():
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(64)(l_gru)
    preds = Dense(NO_CLASSES, activation='softmax')(l_att)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print("model fitting - attention GRU network")
    model.summary()
    model.fit(x_train, y_train,
              nb_epoch=15, batch_size=64)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-LSTM Attention Precision =", precisions)
    print("Classify Glove Bi-LSTM Attention Recall=", recall)
    print("Classify Glove Bi-LSTM Attention F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, NO_CLASSES))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


def biLSTMAttlayer():
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(64)(l_gru)
    preds = Dense(NO_CLASSES, activation='softmax')(l_att)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print("model fitting - attention GRU network")
    model.summary()
    model.fit(x_train, y_train,
              nb_epoch=15, batch_size=64)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-LSTM Attention Precision =", precisions)
    print("Classify Glove Bi-LSTM Attention Recall=", recall)
    print("Classify Glove Bi-LSTM Attention F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


def biLSTMAttDlayer():
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(64)(l_gru)
    l_att = Dense(256, activation="relu")(l_att)
    l_att = Dropout(0.25)(l_att)
    preds = Dense(NO_CLASSES, activation='softmax')(l_att)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    print("model fitting - attention GRU network")
    model.summary()
    model.fit(x_train, y_train,
              nb_epoch=15, batch_size=64)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-GRU Attention Precision =", precisions)
    print("Classify Glove Bi-GRU Attention Recall=", recall)
    print("Classify Glove Bi-GRU Attention F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, NO_CLASSES))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


def cnnmodel():

    model = Sequential()
    model.add(layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train,
                        nb_epoch=15, batch_size=64,
                        validation_data=(x_test, y_test))


    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)


    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r == 1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-LSTM Precision =", precisions)
    print("Classify Glove Bi-LSTM Recall=", recall)
    print("Classify Glove Bi-LSTM F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


def autoEncodeDecodeLayer():
    input_dim = x_train.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation="relu",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoder = Dense(64, activation="relu",  activity_regularizer=regularizers.l1(10e-5))(encoder)
    decoder = Dense(128, activation='relu',  activity_regularizer=regularizers.l1(10e-5))(decoder)
    decoder = Dense(256, activation='relu',  activity_regularizer=regularizers.l1(10e-5))(decoder)
    decoder = Dense(NO_CLASSES, activation='softmax')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    nb_epoch = 15
    batch_size = 128
    autoencoder.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['acc'])

    model = autoencoder.fit(x_train, y_train,
                              epochs=nb_epoch,
                              batch_size=batch_size)

    scores = autoencoder.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    output_test = autoencoder.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r == 1)[0][0] for r in y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify AutoEncoder Precision =", precisions)
    print("Classify AutoEncoder Recall=", recall)
    print("Classify AutoEncoder F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


if __name__ == '__main__':
    autoEncodeDecodeLayer()
    biLSTMAttDlayer()
    biLSTM()
    biGRUAttlayer()
    biLSTMAttlayer()
    cnnmodel()