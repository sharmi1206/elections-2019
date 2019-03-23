import nltk
from nltk.tokenize import *
from nltk.util import ngrams
from nltk.classify import *
import preprocessor as p
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from analysis.tweetprocessor import processMoods as md
from sklearn.linear_model import  SGDClassifier
import nltk
from collections import defaultdict
import datetime
import itertools
import os

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


def plot_precision_recall(precision, recall, f1_score, text):
    num_classes = 8
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    labels = ['anger', 'arousal', 'dominance', 'faith',  'fear', 'joy', 'neutral', 'sadness']
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    count = 0
    for f_score in f1_score:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=colors[count], alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        count = count + 1

    for i in range(num_classes):
        l, = plt.plot(recall[i], precision[i], color=colors[i], lw=2)
        plt.legend(l, labels[i])
        lines.append(l)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of Multi-class Moods for' + text + 'using FastText')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()



def label_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(le_name_mapping)
    return integer_encoded

list_full_data = []

train_files = ['LokShobaElc2019BJP-moods-train.csv', 'LokShobaElc2019Cong-moods-train.csv']
test_files = ['LokShobaElc2019BJP-moods-test.csv', 'LokShobaElc2019Cong-moods-test.csv']
count = 0

def precision(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of test values that appear in the reference set.
    In particular, return card(``reference`` intersection ``test``)/card(``test``).
    If ``test`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    if not hasattr(reference, 'intersection') or not hasattr(test, 'intersection'):
        raise TypeError('reference and test should be sets')

    if len(test) == 0:
        return None
    else:
        return len(reference.intersection(test)) / len(test)


def recall(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of reference values that appear in the test set.
    In particular, return card(``reference`` intersection ``test``)/card(``reference``).
    If ``reference`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    if not hasattr(reference, 'intersection') or not hasattr(test, 'intersection'):
        raise TypeError('reference and test should be sets')

    if len(reference) == 0:
        return None
    else:
        return len(reference.intersection(test)) / len(reference)


def f_measure(reference, test, alpha=0.5):
    """
    Given a set of reference values and a set of test values, return
    the f-measure of the test values, when compared against the
    reference values.  The f-measure is the harmonic mean of the
    ``precision`` and ``recall``, weighted by ``alpha``.  In particular,
    given the precision *p* and recall *r* defined by:

    - *p* = card(``reference`` intersection ``test``)/card(``test``)
    - *r* = card(``reference`` intersection ``test``)/card(``reference``)

    The f-measure is:

    - *1/(alpha/p + (1-alpha)/r)*

    If either ``reference`` or ``test`` is empty, then ``f_measure``
    returns None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    p = precision(reference, test)
    r = recall(reference, test)
    if p is None or r is None:
        return None
    if p == 0 or r == 0:
        return 0
    return 1.0 / (alpha / p + (1 - alpha) / r)


def document_features(document):
    features = {}
    word_list = document['tokenized_tweet']
    if 'tokenized_tweet' in document:
        del document['tokenized_tweet']

    for key in document.keys():
        features[key] = document[key]
    return features


def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


def get_cm_params(classifier, test_set):
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    labels = []
    tests = []
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)
    return labels, tests



def classifyAllTweets():
    count = 0
    for tr_file in train_files:
        l_train_df = pd.read_csv(
            filePath + "/train/sentiments/train-dataset/" + tr_file).dropna()
        l_test_df = pd.read_csv(
            filePath + "/train/sentiments/test-dataset/" + test_files[count]).dropna()

        count = count + 1
        list_full_data.append(l_train_df)
        list_full_data.append(l_test_df)

        fields = ['tweet', 'mood']
        combined_df = pd.concat(list_full_data, axis=0, ignore_index=True)
        data = combined_df.ix[:, ['tweet', 'mood', 'compound', 'pos', 'neg', 'neu', 'polarity', 'retweet_count']]
        data_sentiment = label_encode(combined_df['mood'])
        n = len(data)

        data['tokenized_tweet'] = data.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)

        print('Preprocessing.....')
        for i, row in data.iterrows():
            s = word_tokenize(row[0])
            s = nltk.pos_tag(s)
            data.set_value(i, 'tweet', s)
            data.set_value(i, 'mood', data_sentiment[i])

        print('Creating Data.....')
        df = pd.DataFrame(0, index=range(n),
                          columns=['JUM', 'FUM', 'SUM', 'AUM', 'DUM', 'EUM', 'NUM', 'TUM', 'JBM', 'FBM', 'SBM', 'ABM',
                                   'DBM', 'EBM',
                                   'NBM', 'TBM', 'JTM', 'FTM', 'STM', 'ATM', 'DTM', 'ETM', 'NTM', 'TTM'])



        # Unigram
        print('Caluculating Unigram.....')
        for index, row in data.iterrows():
            JUM = 0
            FUM = 0
            SUM = 0
            AUM = 0
            DUM = 0
            EUM = 0
            NUM = 0
            TUM = 0
            add = 1
            word_mood = []
            for i in range(len(row[0])):
                try:
                    word_mood.append(row[0][i][0])
                    processed_mood = md.get_n_gram_mood(word_mood)
                    if (processed_mood == 'sadness'):
                        SUM = SUM + 1
                    elif (processed_mood == 'joy'):
                        JUM = JUM + 1
                    elif (processed_mood == 'faith'):
                        FUM = FUM + 1
                    elif (processed_mood == 'neutral'):
                        NUM = NUM + 1
                    elif (processed_mood == 'dominance'):
                        DUM = DUM + 1
                    elif (processed_mood == 'arousal'):
                        EUM = EUM + 1
                    elif (processed_mood == 'fear'):
                        TUM = TUM + 1
                    elif (processed_mood == 'anger'):
                        AUM = AUM + 1
                except:
                    continue

                df.set_value(index, 'SUM', SUM)
                df.set_value(index, 'JUM', JUM)
                df.set_value(index, 'FUM', FUM)
                df.set_value(index, 'NUM', NUM)
                df.set_value(index, 'DUM', DUM)
                df.set_value(index, 'EUM', EUM)
                df.set_value(index, 'TUM', TUM)
                df.set_value(index, 'AUM', AUM)

        #Bi-gram
        print('Caluculating Bigram.....')
        for index, row in data.iterrows():
            JBM = 0
            FBM = 0
            SBM = 0
            ABM = 0
            DBM = 0
            EBM = 0
            NBM = 0
            TBM = 0
            add = 1
            word_mood = []
            bigram = ngrams(row[0], 2)
            for pair in bigram:

                word_mood.append(pair[0][0])
                word_mood.append(pair[1][0])
                processed_mood = md.get_n_gram_mood(word_mood)
                # print (processed_mood)
                if (processed_mood == 'sadness'):
                    SBM = SBM + 1
                elif (processed_mood == 'joy'):
                    JBM = JBM + 1
                elif (processed_mood == 'faith'):
                    FBM = FBM + 1
                elif (processed_mood == 'neutral'):
                    NBM = NBM + 1
                elif (processed_mood == 'dominance'):
                    DBM = DBM + 1
                elif (processed_mood == 'arousal'):
                    EBM = EBM + 1
                elif (processed_mood == 'fear'):
                    TBM = TBM + 1
                elif (processed_mood == 'anger'):
                    ABM = ABM + 1

                df.set_value(index, 'SBM', SBM)
                df.set_value(index, 'JBM', JBM)
                df.set_value(index, 'FBM', FBM)
                df.set_value(index, 'NBM', NBM)
                df.set_value(index, 'DBM', DBM)
                df.set_value(index, 'EBM', EBM)
                df.set_value(index, 'TBM', TBM)
                df.set_value(index, 'ABM', ABM)


        #Trigram
        print('Caluculating Trigram.....')
        for index, row in data.iterrows():
            JTM = 0
            FTM = 0
            STM = 0
            ATM = 0
            DTM = 0
            ETM = 0
            TTM = 0
            NTM = 0
            add = 1
            word_mood = []
            trigram = ngrams(row[0], 3)
            for pair in trigram:

                word_mood.append(pair[1][0])
                word_mood.append(pair[0][0])
                word_mood.append(pair[2][0])
                processed_mood = md.get_n_gram_mood(word_mood)
                # print (processed_mood)
                if (processed_mood == 'sadness'):
                    STM = STM + 1
                elif (processed_mood == 'joy'):
                    JTM = JTM + 1
                elif (processed_mood == 'faith'):
                    FTM = FTM + 1
                elif (processed_mood == 'neutral'):
                    NTM = NTM + 1
                elif (processed_mood == 'dominance'):
                    DTM = DTM + 1
                elif (processed_mood == 'arousal'):
                    ETM = ETM + 1
                elif (processed_mood == 'fear'):
                    TTM = TTM + 1
                elif (processed_mood == 'anger'):
                    ATM = ATM + 1

                df.set_value(index, 'STM', STM)
                df.set_value(index, 'JTM', JTM)
                df.set_value(index, 'FTM', FTM)
                df.set_value(index, 'NTM', NTM)
                df.set_value(index, 'DTM', DTM)
                df.set_value(index, 'ETM', ETM)
                df.set_value(index, 'TTM', TTM)
                df.set_value(index, 'ATM', ATM)

        sentiment_tuple_list = []
        df['tokenized_tweet'] = data['tokenized_tweet']
        df['compound'] = data['compound']
        df['pos'] = data['pos']
        df['neg'] = data['neg']
        df['polarity'] = data['polarity']
        df['retweet_count'] = data['retweet_count']

        for i in range(0, len(df)):
            sentiment_tuple = (df.to_dict('records')[i], data['mood'].values[i])
            sentiment_tuple_list.append(sentiment_tuple)

        featuresets = [(document_features(d), c) for (d, c) in sentiment_tuple_list]
        train_set, test_set = featuresets[100:], featuresets[:100]

        now = datetime.datetime.now()
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        classifier.show_most_informative_features(20)
        print('accuracy by using Naive Bayes:', nltk.classify.util.accuracy(classifier, test_set))

        labels, tests = get_cm_params(classifier, test_set)
        print('Naive Bayes Precision', precision(set(labels), set(tests)))
        print('Naive Bayes Recall', recall(set(labels), set(tests)))
        print('Naive Bayes F_Score', f_measure(set(labels), set(tests)))
        print('Naive Bayes Confusion Matrix' , nltk.ConfusionMatrix(labels, tests))
        print('Naive Bayes Classification Report', classification_report(labels, tests))

        print('SVM Linear Kernel.....')
        now = datetime.datetime.now()
        classifier = SklearnClassifier(SVC(kernel='linear')).train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SVM:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('Linear SVM Precision', precision(set(labels), set(tests)))
        print('Linear SVM Recall', recall(set(labels), set(tests)))
        print('Linear SVM F_Score', f_measure(set(labels), set(tests)))
        print('Linear SVM Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        print('Linear SVM Classification Report', classification_report(labels, tests))

        print('SVM RBF Kernel.....')
        now = datetime.datetime.now()
        classifier = SklearnClassifier(SVC(kernel='rbf')).train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SVM:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('RBF SVM Precision', precision(set(labels), set(tests)))
        print('RBF SVM Recall', recall(set(labels), set(tests)))
        print('RBF SVM F_Score', f_measure(set(labels), set(tests)))
        print('RBF SVM Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        print('RBF SVM Classification Report', classification_report(labels, tests))

        print('SGD Classifier.....')
        sgd = SklearnClassifier(SGDClassifier(loss='log'))
        now = datetime.datetime.now()
        sgd.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SGDClassifier:', nltk.classify.util.accuracy(sgd, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('SGD Precision', precision(set(labels), set(tests)))
        print('SGD Recall', recall(set(labels), set(tests)))
        print('SGD F_Score', f_measure(set(labels), set(tests)))
        print('SGD Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        print('SGD Classification Report', classification_report(labels, tests))

        print('Decision Tree.....')
        now = datetime.datetime.now()
        classifier = DecisionTreeClassifier.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        # classifier.best_binary_stump()
        print('Accuracy by using Decision Tree:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print(nltk.ConfusionMatrix(labels, tests))
        print('Decision Tree Precision', precision(set(labels), set(tests)))
        print('Decision Tree Recall', recall(set(labels), set(tests)))
        print('Decision Tree F_Score', f_measure(set(labels), set(tests)))
        print('Decision Tree Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        print('Decision Tree Classification Report', classification_report(labels, tests))

        print('Training and Testing by using Max Entropy.....')
        now = datetime.datetime.now()
        classifier = MaxentClassifier.train(train_set, max_iter=10)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        classifier.show_most_informative_features()
        print('accuracy by using Max Entropy:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print(nltk.ConfusionMatrix(labels, tests))
        print('Max Entropy Precision', precision(set(labels), set(tests)))
        print('Max Entropy Recall', recall(set(labels), set(tests)))
        print('Max Entropy F_Score', f_measure(set(labels), set(tests)))
        print('Max Entropy Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        print('Max Entropy Classification Report', classification_report(labels, tests))

def classifyPolarizedTweets():
    count = 0
    for tr_file in train_files:
        l_train_df = pd.read_csv(
            filePath + "/train/sentiments/train-dataset/" + tr_file).dropna()
        l_test_df = pd.read_csv(
            filePath + "/train/sentiments/test-dataset/" + test_files[count]).dropna()

        count = count + 1

        list_full_data.append(l_train_df)
        list_full_data.append(l_test_df)

        fields = ['tweet', 'mood']
        combined_df = pd.concat(list_full_data, axis=0, ignore_index=True)
        data = combined_df.ix[:, ['tweet', 'mood']]
        data_sentiment = label_encode(combined_df['mood'])
        n = len(data)

        #Preprocessing data by nltk.tokenize,nltk.pos_tag and preprocessor package
        print('Preprocessing.....')
        for i, row in data.iterrows():
            s = word_tokenize(row[0])
            s = nltk.pos_tag(s)
            data.set_value(i, 'tweet', s)
            data.set_value(i, 'mood', data_sentiment[i])

        df = pd.DataFrame(0, index=range(n),
                          columns=['JUM', 'FUM', 'SUM', 'AUM', 'DUM', 'EUM', 'NUM', 'JBM', 'FBM', 'SBM', 'ABM',
                                   'DBM', 'EBM', 'NBM', 'JTM', 'FTM', 'STM', 'ATM', 'DTM', 'ETM', 'NTM', 'y'])
        df.y = data.mood

        #Unigram
        print('Caluculating Unigram.....')
        for index, row in data.iterrows():
            JUM = 0
            FUM = 0
            SUM = 0
            AUM = 0
            DUM = 0
            EUM = 0
            NUM = 0
            TUM = 0
            add = 1
            word_mood = []
            for i in range(len(row[0])):
                if row[0][i][1] == 'JJ' or row[0][i][1] == 'RB':
                    try:
                        word_mood.append(row[0][i][0])
                        processed_mood = md.get_n_gram_mood(word_mood)
                        if (processed_mood == 'sadness'):
                            SUM = SUM + 1
                        elif (processed_mood == 'joy'):
                            JUM = JUM + 1
                        elif (processed_mood == 'faith'):
                            FUM = FUM + 1
                        elif (processed_mood == 'neutral'):
                            NUM = NUM + 1
                        elif (processed_mood == 'dominance'):
                            DUM = DUM + 1
                        elif (processed_mood == 'arousal'):
                            EUM = EUM + 1
                        elif (processed_mood == 'fear'):
                            TUM = TUM + 1
                        elif (processed_mood == 'anger'):
                            AUM = AUM + 1
                    except:
                        continue
            # print (i,PUM,NUM)
            df.set_value(index, 'SUM', SUM)
            df.set_value(index, 'JUM', JUM)
            df.set_value(index, 'FUM', FUM)
            df.set_value(index, 'NUM', NUM)
            df.set_value(index, 'DUM', DUM)
            df.set_value(index, 'EUM', EUM)
            df.set_value(index, 'TUM', TUM)
            df.set_value(index, 'AUM', AUM)

        print('Caluculating Bigram.....')
        for index, row in data.iterrows():
            JBM = 0
            FBM = 0
            SBM = 0
            ABM = 0
            DBM = 0
            EBM = 0
            NBM = 0
            TBM = 0
            word_mood = []
            bigram = ngrams(row[0], 2)
            for pair in bigram:
                if ((pair[0][1] == 'JJ' or pair[0][1] == 'RB') or (pair[1][1] == 'JJ' or pair[1][1] == 'RB')):
                    word_mood.append(pair[0][0])
                    word_mood.append(pair[1][0])
                    processed_mood = md.get_n_gram_mood(word_mood)
                    if (processed_mood == 'sadness'):
                        SBM = SBM + 1
                    elif (processed_mood == 'joy'):
                        JBM = JBM + 1
                    elif (processed_mood == 'faith'):
                        FBM = FBM + 1
                    elif (processed_mood == 'neutral'):
                        NBM = NBM + 1
                    elif (processed_mood == 'dominance'):
                        DBM = DBM + 1
                    elif (processed_mood == 'arousal'):
                        EBM = EBM + 1
                    elif (processed_mood == 'fear'):
                        TBM = TBM + 1
                    elif (processed_mood == 'anger'):
                        ABM = ABM + 1

                df.set_value(index, 'SBM', SBM)
                df.set_value(index, 'JBM', JBM)
                df.set_value(index, 'FBM', FBM)
                df.set_value(index, 'NBM', NBM)
                df.set_value(index, 'DBM', DBM)
                df.set_value(index, 'EBM', EBM)
                df.set_value(index, 'TBM', TBM)
                df.set_value(index, 'ABM', ABM)

        # Trigram
        print('Caluculating Trigram.....')
        for index, row in data.iterrows():
            JTM = 0
            FTM = 0
            STM = 0
            ATM = 0
            DTM = 0
            ETM = 0
            TTM = 0
            NTM = 0
            word_mood = []
            trigram = ngrams(row[0], 3)
            for pair in trigram:
                if ((pair[0][1] == 'JJ' or pair[0][1] == 'RB') or (
                        pair[1][1] == 'JJ' or pair[1][1] == 'RB') or (
                        pair[2][1] == 'JJ' or pair[2][1] == 'RB')):
                    word_mood.append(pair[1][0])
                    word_mood.append(pair[0][0])
                    word_mood.append(pair[2][0])
                    processed_mood = md.get_n_gram_mood(word_mood)
                    # print (processed_mood)
                    if (processed_mood == 'sadness'):
                        STM = STM + 1
                    elif (processed_mood == 'joy'):
                        JTM = JTM + 1
                    elif (processed_mood == 'faith'):
                        FTM = FTM + 1
                    elif (processed_mood == 'neutral'):
                        NTM = NTM + 1
                    elif (processed_mood == 'dominance'):
                        DTM = DTM + 1
                    elif (processed_mood == 'arousal'):
                        ETM = ETM + 1
                    elif (processed_mood == 'fear'):
                        TTM = TTM + 1
                    elif (processed_mood == 'anger'):
                        ATM = ATM + 1

                df.set_value(index, 'STM', STM)
                df.set_value(index, 'JTM', JTM)
                df.set_value(index, 'FTM', FTM)
                df.set_value(index, 'NTM', NTM)
                df.set_value(index, 'DTM', DTM)
                df.set_value(index, 'ETM', ETM)
                df.set_value(index, 'TTM', TTM)
                df.set_value(index, 'ATM', ATM)

        print('Spliting data.....')
        featuresets = [(row[:23], row[24]) for index, row in df.iterrows()]
        split = int(n * 0.8)
        train_set = featuresets[:split]
        test_set = featuresets[split:]

        now = datetime.datetime.now()
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        classifier.show_most_informative_features(20)
        print('accuracy by using Naive Bayes:', nltk.classify.util.accuracy(classifier, test_set))

        labels, tests = get_cm_params(classifier, test_set)
        print('Naive Bayes Precision', precision(set(labels), set(tests)))
        print('Naive Bayes Recall', recall(set(labels), set(tests)))
        print('Naive Bayes F_Score', f_measure(set(labels), set(tests)))
        print('Naive Bayes Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        #print('Naive Bayes Classification Report', classification_report(labels, tests))

        print('SVM Linear Kernel.....')
        now = datetime.datetime.now()
        classifier = SklearnClassifier(SVC(kernel='linear')).train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SVM:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('Linear SVM Precision', precision(set(labels), set(tests)))
        print('Linear SVM Recall', recall(set(labels), set(tests)))
        print('Linear SVM F_Score', f_measure(set(labels), set(tests)))
        print('Linear SVM Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        #print('Linear SVM Classification Report', classification_report(labels, tests))

        print('SVM RBF Kernel.....')
        now = datetime.datetime.now()
        classifier = SklearnClassifier(SVC(kernel='rbf')).train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SVM:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('RBF SVM Precision', precision(set(labels), set(tests)))
        print('RBF SVM Recall', recall(set(labels), set(tests)))
        print('RBF SVM F_Score', f_measure(set(labels), set(tests)))
        print('RBF SVM Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        #print('RBF SVM Classification Report', classification_report(labels, tests))

        print('SGD Classifier.....')
        sgd = SklearnClassifier(SGDClassifier(loss='log'))
        now = datetime.datetime.now()
        sgd.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        print('accuracy by using SGDClassifier:', nltk.classify.util.accuracy(sgd, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print('SGD Precision', precision(set(labels), set(tests)))
        print('SGD Recall', recall(set(labels), set(tests)))
        print('SGD F_Score', f_measure(set(labels), set(tests)))
        print('SGD Confusion Matrix', nltk.ConfusionMatrix(labels, tests))
        #print('SGD Classification Report', classification_report(labels, tests))

        print('Decision Tree.....')
        now = datetime.datetime.now()
        classifier = DecisionTreeClassifier.train(train_set)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        # classifier.best_binary_stump()
        print('Accuracy by using Decision Tree:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print(nltk.ConfusionMatrix(labels, tests))
        print('Decision Tree Precision', precision(set(labels), set(tests)))
        print('Decision Tree Recall', recall(set(labels), set(tests)))
        print('Decision Tree F_Score', f_measure(set(labels), set(tests)))
        print('Decision Tree Confusion Matrix', nltk.ConfusionMatrix(labels, tests))

        print('Training and Testing by using Max Entropy.....')
        now = datetime.datetime.now()
        classifier = MaxentClassifier.train(train_set, max_iter=10)
        elapsed = (datetime.datetime.now() - now)
        print(elapsed.microseconds)
        print(elapsed.seconds)
        classifier.show_most_informative_features()
        print('accuracy by using Max Entropy:', nltk.classify.util.accuracy(classifier, test_set))
        labels, tests = get_cm_params(classifier, test_set)
        print(nltk.ConfusionMatrix(labels, tests))
        print('Max Entropy Precision', precision(set(labels), set(tests)))
        print('Max Entropy Recall', recall(set(labels), set(tests)))
        print('Max Entropy F_Score', f_measure(set(labels), set(tests)))
        print('Max Entropy Confusion Matrix', nltk.ConfusionMatrix(labels, tests))


classifyAllTweets()
classifyPolarizedTweets()