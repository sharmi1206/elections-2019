import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

import os
import pandas as pd
import utils as ul
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as mplt

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

tfidf_vect, tfidf_vect_ngram, tfidf_vect_ngram_chars, vectorizer = ul.get_vectorize_ngrams()

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
    mplt.savefig(filePath + "/figs/congress/bagging/"+ figname + ".png")  #save the figs


def apply_rf_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = RandomForestClassifier(n_estimators=500,
                                max_features=0.25,
                                criterion="entropy",
                                class_weight="balanced")
    clf.fit(X, training_set_y)

    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore_1 = metrics.accuracy_score(test_set_y, pred)
    print("RandomForest Accuracy" , pscore_1)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("RandomForest Precision =", precisions)
    print("RandomForest Recall =", recall)
    print("RandomForest F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with RandomForestClassifier', 'rf-cv')

    clf = RandomForestClassifier(n_estimators=500,
                                max_features=0.25,
                                criterion="entropy",
                                class_weight="balanced")
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore_2 = metrics.accuracy_score(test_set_y, pred)
    print("RandomForest Accuracy" , pscore_2)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("RandomForest Precision =", precisions)
    print("RandomForest Recall =", recall)
    print("RandomForest F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with RandomForestClassifier', 'rf-tdidf')



    text_clf = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('rf', RandomForestClassifier(n_estimators=500,
                                max_features=0.25,
                                criterion="entropy",
                                class_weight="balanced"))])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore_3 = metrics.accuracy_score(test_set_y, pred)
    print("RandomForest Accuracy" , pscore_3)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("RandomForest Precision =", precisions)
    print("RandomForest Recall =", recall)
    print("RandomForest F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector & TfIdf with RandomForestClassifier', 'rf-pip')


def apply_extratree_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = ExtraTreesClassifier(criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=0.25)

    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    print("ExtraTrees-count vector" + str(pred))
    pscore_1 = metrics.accuracy_score(test_set_y, pred)
    print("ExtraTrees Accuracy" , pscore_1)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("ExtraTrees Precision =", precisions)
    print("ExtraTrees Recall =", recall)
    print("ExtraTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with ExtraTreesClassifier', 'et-cv')



    clf = ExtraTreesClassifier(criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=0.25)
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    print("ExtraTrees - tfidf" + str(pred))
    pscore_2 = metrics.accuracy_score(test_set_y, pred)
    print("ExtraTrees Accuracy" , pscore_2)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("ExtraTrees Precision =", precisions)
    print("ExtraTrees Recall =", recall)
    print("ExtraTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with ExtraTreesClassifier', 'et-tdif')

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('eb', ExtraTreesClassifier(criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=0.25))])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore_3 = metrics.accuracy_score(test_set_y, pred)
    print("ExtraTrees Accuracy" , pscore_3)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("ExtraTrees Precision =", precisions)
    print("ExtraTrees Recall =", recall)
    print("ExtraTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector & TfIdf with ExtraTreesClassifier', 'et-pip')


def apply_decisiontree_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = DecisionTreeClassifier()
    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore_1 = metrics.accuracy_score(test_set_y, pred)
    print("DecisionTrees Accuracy" , pscore_1)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("DecisionTrees Precision =", precisions)
    print("DecisionTrees Recall =", recall)
    print("DecisionTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with DecisionTreeClassifier', 'dt-cv')


    clf = DecisionTreeClassifier()
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore_2 = metrics.accuracy_score(test_set_y, pred)
    print("DecisionTrees Accuracy" , pscore_2)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("DecisionTrees Precision =", precisions)
    print("DecisionTrees Recall =", recall)
    print("DecisionTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with DecisionTreeClassifier', 'dt-tdif')


    text_clf = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('eb', DecisionTreeClassifier())])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore_3 = metrics.accuracy_score(test_set_y, pred)
    print("DecisionTrees Accuracy" , pscore_3)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("DecisionTrees Precision =", precisions)
    print("DecisionTrees Recall =", recall)
    print("DecisionTrees F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector & TfIdf with DecisionTreeClassifier', 'dt-pip')


def apply_clf_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = BaggingClassifier(n_estimators =25,
                 max_features=0.25)
    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore_1 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_1)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with BaggingClassifier', 'bg-cv')


    clf = BaggingClassifier(n_estimators =25,
                 max_features=0.25)
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore_2 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_2)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)


    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with BaggingClassifier', 'bg-tdif')

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('pipbg', BaggingClassifier(n_estimators =25,
                 max_features=0.25))])


    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore_3 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_3)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector & TfIdf with BaggingClassifier on RandomForest', 'bg-pip')


def apply_clf_rf_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    rf = RandomForestClassifier(n_estimators=500,
                                 max_features=0.25,
                                 criterion="entropy",
                                 class_weight="balanced")

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = BaggingClassifier(base_estimator = rf, n_estimators =25,
                 max_features=0.25)
    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore_1 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_1)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :CountVector with BaggingClassifier on RandomForest', 'bg-rf-cv')


    clf = BaggingClassifier(base_estimator = rf, n_estimators =25,
                 max_features=0.25)
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore_2 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_2)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)


    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :TfIdf with BaggingClassifier on RandomForest', 'bg-rf-tdif')

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('pipbg', BaggingClassifier(n_estimators =25,
                 max_features=0.25))])


    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore_3 = metrics.accuracy_score(test_set_y, pred)
    print("Bagging Accuracy" , pscore_3)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Bagging Precision =", precisions)
    print("Bagging Recall =", recall)
    print("Bagging F1 score =", f1_score)
    plot_precision_recall(precisions, recall, f1_score, ' ' + party_name + ' :Count Vector & TfIdf with BaggingClassifier on RandomForest', 'bg-rf-pip')



if __name__ == '__main__':

    tain_path = filePath + "/train/sentiments/train-dataset/"
    test_path = filePath + "/train/sentiments/test-dataset/"
    train_file_names = os.listdir(tain_path)
    test_file_names = os.listdir(test_path)
    party_names = ['Bjp', 'Congress', 'Bjp-Congress', 'Neutral']
    for i in range(0, len(train_file_names)):
        train = pd.read_csv(tain_path+ train_file_names[i]).dropna()
        test = pd.read_csv(test_path+ test_file_names[i]).dropna()

        print("Processing file.." + train_file_names[i])

        training_set_x = train['tweet'].values
        training_set_y = ul.label_encode(train['mood'])
        test_set_x = test['tweet'].values
        test_set_y = ul.label_encode(test['mood'])
        apply_rf_bagging(training_set_x, training_set_y, test_set_x, test_set_y, party_names[i])
        apply_clf_bagging(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])
        apply_extratree_bagging(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])
        apply_decisiontree_bagging(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])
        apply_clf_rf_bagging(training_set_x, training_set_y, test_set_x,
                                                                    test_set_y,  party_names[i])


