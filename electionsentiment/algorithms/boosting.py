import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import  XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import datetime
from sklearn.preprocessing import label_binarize
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import utils as ul

import os
from sklearn.pipeline import Pipeline
import lightgbm as lgb

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


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
    plt.title('Precision-Recall Curve for ' + text, fontsize =6)
    plt.legend(lines, full_label, loc=(0.6, .7), prop=dict(size=8))
    plt.savefig(filePath + "/figs/bjp/boosting/"+ figname + ".png")

def timer(start_time=None):
    if not start_time:
        start_time = datetime.datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def apply_gradient_boosting(training_set_x, training_set_y, test_set_x, test_set_y, party_name):


    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = GradientBoostingClassifier(n_estimators =100, learning_rate =0.1, max_depth=6, min_samples_leaf =1, max_features=1.0)
    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("GradientBoosting Precision =", precisions)
    print("GradientBoosting Recall =", recall)
    print("GradientBoosting F1 score =", f1_score)
    print("Gradient Boosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector with GradientBoostClassifier', 'gb-count')

    clf = GradientBoostingClassifier(n_estimators =100, learning_rate =0.1, max_depth=6, min_samples_leaf =1, max_features=1.0)
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("GradientBoosting Precision =", precisions)
    print("GradientBoosting Recall =", recall)
    print("GradientBoosting F1 score =", f1_score)
    print("Gradient Boosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :TfIdff with GradientBoostClassifier', 'gb-tfidf')

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('gb', GradientBoostingClassifier(n_estimators =100, learning_rate =0.1, max_depth=6, min_samples_leaf =1, max_features=1.0))])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("GradientBoosting Precision =", precisions)
    print("GradientBoosting Recall =", recall)
    print("GradientBoosting F1 score =", f1_score)
    print("Gradient Boosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector & TfIdf with GradientBoostClassifier', 'gb-pip')


def apply_lightgbm_boosting(training_set_x, training_set_y, test_set_x, test_set_y, party_name):
    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'multi-class',
          n_jobs = 3,
          silent = True,
          max_depth = 4,colsample_bytree=0.66, subsample = 0.75, num_leaves =16)

    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Light GBM Precision =", precisions)
    print("Light GBM Recall =", recall)
    print("Light GBM F1 score =", f1_score)
    print("Light GBM Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector with LightGBMClassifier', 'lgb-count')

    clf = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='multi-class',
                             n_jobs=3,
                             silent=True,
                             max_depth=4, colsample_bytree=0.66, subsample=0.75, num_leaves=16)

    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Light GBM Precision =", precisions)
    print("Light GBM Recall =", recall)
    print("Light GBM F1 score =", f1_score)
    print("Light GBM Boosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :TfIdf with LightGBMClassifier', 'lgb-tfidf')

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('lgb', lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='multi-class',
                             n_jobs=3,
                             silent=True,max_depth = 4,colsample_bytree=0.66, subsample = 0.75, num_leaves =16))])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("Light GBM Precision =", precisions)
    print("Light GBM Recall =", recall)
    print("Light GBM F1 score =", f1_score)
    print("Light GBM Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector & TfIdf with LightGBMClassifier', 'lgb-pip')




def apply_xg_boosting(training_set_x, training_set_y, test_set_x, test_set_y, party_name):
    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    xgb = XGBClassifier(max_depth=2, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 booster='gbtree')

    xgb.fit(X, training_set_y)
    pred = xgb.predict(vectorizer.transform(test_set_x).toarray())
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("XgBoost Precision =", precisions)
    print("XgBoost Recall =", recall)
    print("XgBoost F1 score =", f1_score)
    print("XgBoost Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector with XgBoostClassifier', 'xgb-count')

    xgb = XGBClassifier(max_depth=2, learning_rate=0.1,
                        n_estimators=100, silent=True,
                        booster='gbtree')

    xgb.fit(xtrain_tfidf, training_set_y)
    pred = xgb.predict(xvalid_tfidf)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("XgBoost Precision =", precisions)
    print("XgBoost Recall =", recall)
    print("XgBoost F1 score =", f1_score)
    print("XgBoost Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :TfIdf with XgBoostClassifier', 'xgb-tfidf')

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('xgb', XGBClassifier(max_depth=2, learning_rate=0.1,
                        n_estimators=100, silent=True,
                        booster='gbtree'))])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("XgBoost Precision =", precisions)
    print("XgBoost Recall =", recall)
    print("XgBoost F1 score =", f1_score)
    print("XgBoost Boosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector & TfIdf with XgBoostClassifier', 'xgb-pip')


def apply_ada_boosting(training_set_x, training_set_y, test_set_x, test_set_y, party_name):

    X = vectorizer.fit_transform(training_set_x).toarray()
    tfidf_vect_ngram.fit(training_set_x)
    xtrain_tfidf = tfidf_vect_ngram.transform(training_set_x)
    xvalid_tfidf = tfidf_vect_ngram.transform(test_set_x)

    clf = AdaBoostClassifier()
    clf.fit(X, training_set_y)
    pred = clf.predict(vectorizer.transform(test_set_x).toarray())
    pscore = metrics.accuracy_score(test_set_y, pred)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("AdaBoosting Precision =", precisions)
    print("AdaBoosting Recall =", recall)
    print("AdaBoosting F1 score =", f1_score)
    print("AdaBoosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector with AdaBoostClassifier', 'ab-cv')

    clf = AdaBoostClassifier()
    clf.fit(xtrain_tfidf, training_set_y)
    pred = clf.predict(xvalid_tfidf)
    print("AdaBoost - tfidf" + str(pred))
    pscore = metrics.accuracy_score(test_set_y, pred)
    print(pscore)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("AdaBoosting Precision =", precisions)
    print("AdaBoosting Recall =", recall)
    print("AdaBoosting F1 score =", f1_score)
    print("AdaBoosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :TfIdf with AdaBoostClassifier', 'ab-tfidf')

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('ab', AdaBoostClassifier())])
    text_clf.fit(training_set_x, training_set_y)
    pred = text_clf.predict(test_set_x)
    pscore = metrics.accuracy_score(test_set_y, pred)
    print("AdaBoost - pipeline count vector + tdiff" + str(pred))
    print(pscore)
    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(test_set_y, pred)
    print("AdaBoosting Precision =", precisions)
    print("AdaBoosting Recall =", recall)
    print("AdaBoosting F1 score =", f1_score)
    print("AdaBoosting Accuracy=", pscore)
    plot_precision_recall(precisions, recall, f1_score,
                          ' ' + party_name + ' :CountVector & TfIdf with AdaBoostClassifier', 'ab-pip')




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
        apply_xg_boosting(training_set_x, training_set_y, test_set_x, test_set_y, party_names[i])
        apply_gradient_boosting(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])
        apply_ada_boosting(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])
        apply_lightgbm_boosting(training_set_x, training_set_y, test_set_x, test_set_y,  party_names[i])

