import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from gensim.models.word2vec import Word2Vec

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# Set random seed (for reproducibility)
np.random.seed(1000)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


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


import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]




def predict_mood_cnn(combined_df):

    train_mood = one_hot_encode(combined_df['mood'])
    corpus = combined_df['tweet'].values
    all_labels = train_mood

    print('Corpus size: {}'.format(len(corpus)))

    # Tokenize and stem
    tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
    stemmer = LancasterStemmer()

    tokenized_corpus = []

    for i, tweet in enumerate(corpus):
        tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
        tokenized_corpus.append(tokens)

    # Gensim Word2Vec model
    vector_size = 512
    window_size = 10
    num_words = 20000

    # Create Word2Vec
    word2vec = Word2Vec(sentences=tokenized_corpus,
                        size=vector_size,
                        window=window_size,
                        iter=500,
                        seed=300,
                        workers=multiprocessing.cpu_count())

    # Copy word vectors and delete Word2Vec model  and original corpus to save memory
    X_vecs = word2vec.wv
    del word2vec
    del corpus

    # Train subset size (0 < size < len(tokenized_corpus))
    train_size = int(len(l_train_df))

    # Test subset size (0 < size < len(tokenized_corpus) - train_size)
    test_size = int(len(l_test_df)) -1

    # Compute average and max tweet length
    avg_length = 0.0
    max_length = 0

    for tweet in tokenized_corpus:
        if len(tweet) > max_length:
            max_length = len(tweet)
        avg_length += float(len(tweet))

    print('Average tweet length: {}'.format(avg_length / float(len(tokenized_corpus))))
    print('Max tweet length: {}'.format(max_length))

    # Tweet max length (number of tokens)
    max_tweet_length = 100

    # Create train and test sets
    # Generate random indexes
    indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size, replace=False))

    X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
    Y_train = np.zeros((train_size, 8), dtype=np.int32)
    X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
    Y_test = np.zeros((test_size, 8), dtype=np.int32)
    Y_feature_out = []



    try:
        for i, index in enumerate(indexes):
            for t, token in enumerate(tokenized_corpus[index]):
                if t >= max_tweet_length:
                    break

                if token not in X_vecs:
                    continue

                if i < train_size:
                    X_train[i, t, :] = X_vecs[token]
                else:
                    X_test[i - train_size, t, :] = X_vecs[token]

            if i < train_size:
                Y_train[i, :] = all_labels[index].tolist()
            else:
                Y_test[i - train_size, :] = all_labels[index].tolist()

    except Exception as e:
            pass



    # Keras convolutional model
    batch_size = 64
    nb_epochs = 20


    max_features = 1000
    maxlen = 201
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250


    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length, vector_size)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, kernel_size=3, activation='elu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, kernel_size=3, activation='elu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(8, activation='softmax'))



    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=nb_epochs)

    scores = model.evaluate(X_test, Y_test, verbose=0)

    output_test = model.predict(X_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r==1)[0][0] for r in Y_test]
    print(org_y_label)
    results = confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify W2Vec CNN Precision =", precisions)
    print("Classify W2Vec CNN Recall=", recall)
    print("Classify W2Vec CNN F1 Score =", f1_score)


    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0,8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))

if __name__ == '__main__':

    tain_path = filePath + "/train/sentiments/train-dataset/"
    test_path = filePath + "/train/sentiments/test-dataset/"
    train_file_names = os.listdir(tain_path)
    test_file_names = os.listdir(test_path)
    party_names = ['Bjp', 'Congress', 'Bjp-Congress', 'Neutral']
    for i in range(0, len(train_file_names)-2):
        corpusList = []
        labelList = []

        list_train = []

        l_train_df = pd.read_csv(
            tain_path + train_file_names[i]).dropna()
        l_test_df = pd.read_csv(test_path + test_file_names[i]).dropna()

        list_train.append(l_train_df)
        list_train.append(l_test_df)

        combined_df = pd.concat(list_train, axis=0, ignore_index=True)

        predict_mood_cnn(combined_df)