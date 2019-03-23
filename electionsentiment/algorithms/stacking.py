from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import  XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import utils as ul
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import os
import inspect

fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


train = pd.read_csv(filePath + "/train/sentiments/train-dataset/LokShobaElc2019BJP-moods-train.csv").dropna()
test =  pd.read_csv(filePath + "/train/sentiments/test-dataset/LokShobaElc2019BJP-moods-test.csv").dropna()
tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, vectorizer = ul.get_vectorize_ngrams()


def label_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    return integer_encoded


def plot_precision_recall(precision, recall, f1_score, text, figname):
    num_classes = 8
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']
    full_label = []
    labels = ['anger', 'arousal', 'dominance', 'faith',  'fear', 'joy', 'neutral', 'sadness']
    plt.figure(figsize=(7, 8))
    lines = []
    count = 0
    for f_score in f1_score:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color=colors[count], alpha=0.2)
        full_label.append(labels[count] + ',f1={0:0.1f},'.format(f_score) + 'p={0:0.1f},'.format(precision[count]) +  'r={0:0.1f}'.format(recall[count]),)
        count = count + 1

    for i in range(num_classes):
        l, = plt.plot(recall[i], precision[i], color=colors[i], lw=2)
        plt.legend([l], [full_label[i] ], loc=(0.6, .7), prop=dict(size=8))
        lines.append(l)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for' + text, fontsize =6)
    plt.legend(lines, full_label, loc=(0.6, .7), prop=dict(size=8))
    plt.savefig(filePath + "/figs/congress/ensembling/"+ figname + ".png")  #save the figs

training_set_x = train['tweet'].values
training_set_y = label_encode(train['mood'])
test_set_x = test['tweet'].values
test_set_y = label_encode(test['mood'])
X = vectorizer.fit_transform(training_set_x).toarray()


tfidf_vect_ngram.fit(training_set_x)
xtrain_tfidf =  tfidf_vect_ngram.transform(training_set_x)
xvalid_tfidf =  tfidf_vect_ngram.transform(test_set_x)

def apply_ensemble_stacking(training_set_x, training_set_y, test_set_x, test_set_y, party_name):
    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)
    estimators = []
    model0 = LogisticRegression()

    model1 = RandomForestClassifier(n_estimators=500,
                                 max_features=0.25,
                                 criterion="entropy",
                                 class_weight="balanced")

    estimators.append(('lgr', model0))
    estimators.append(('rf', model1))

    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))

    model3 = SVC()
    estimators.append(('svc', model3))

    ensemble = VotingClassifier(estimators, weights=[1, 2, 2, 1])
    ensemble.fit(X, training_set_y)
    pred = ensemble.predict(vectorizer.transform(test_set_x).toarray())
    accuracy = metrics.accuracy_score(test_set_y, pred)
    print("Ensembling=" , accuracy)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Ensembling Precision =", precisions)
    print("Ensembling Recall =", recall)
    print("Ensembling F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with VotingClassifier', 'vt-cv')

    ensemble.fit(xtrain_tfidf, training_set_y)
    pred = ensemble.predict(xvalid_tfidf)
    accuracy = metrics.accuracy_score(test_set_y, pred)
    print("Ensembling=", accuracy)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Ensembling Precision =", precisions)
    print("Ensembling Recall =", recall)
    print("Ensembling F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with VotingClassifier', 'vt-tfidf')

    text_clf = Pipeline(
        [('vect', vectorizer), ('tfidf', TfidfTransformer()), ('vt', VotingClassifier(estimators, weights=[1, 2, 2, 1]))])

    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, pred)
    print("Ensembling=", accuracy)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Ensembling Precision =", precisions)
    print("Ensembling Recall =", recall)
    print("Ensembling F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector & TfIdf with VotingClassifier', 'vt-pip')



if __name__ == '__main__':

    tain_path = filePath + '/train/sentiments/train-dataset/'
    test_path = filePath + '/train/sentiments/test-dataset/'
    party_names = ['Bjp', 'Congress', 'Bjp-Congress', 'Neutral']
    train_file_names = os.listdir(tain_path)
    test_file_names = os.listdir(test_path)
    for i in range(0, len(train_file_names)):
        train = pd.read_csv(tain_path+ train_file_names[i]).dropna()
        test = pd.read_csv(test_path+ test_file_names[i]).dropna()

        print("Processing file.." + train_file_names[i])

        training_set_x = train['tweet'].values
        training_set_y = ul.label_encode(train['mood'])
        test_set_x = test['tweet'].values
        test_set_y = ul.label_encode(test['mood'])
        apply_ensemble_stacking(training_set_x, training_set_y, test_set_x, test_set_y. party_names[i])
