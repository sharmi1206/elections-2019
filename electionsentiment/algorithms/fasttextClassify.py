# Import libraries/functions
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import fastText as ft
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
import os
import datetime

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


df_train = pd.read_csv(filePath + "/train/sentiments/train-dataset/LokShobaElc2019BJP-moods-train.csv")
df_test =  pd.read_csv(filePath + "/train/sentiments/test-dataset/LokShobaElc2019BJP-moods-test.csv")

labelled_mood = '__label__' + df_train.mood
df_train['labels_text'] = labelled_mood
df_test['labels_text'] = '__label__' + df_test.mood
df_train.labels_text = df_train.labels_text.str.cat(df_train.tweet, sep=' ')
num_classes = len(df_test.mood.unique())
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']


df_train.labels_text.to_csv('train.txt', index =False)


# Function to do K-fold CV across different fasttext parameter values
def cross_validate(YX, k, lr, wordNgrams, epoch):
    # Record results
    results = []
    accuracy = []
    pred_filter = []
    org = []
    cross_valid_params = {}
    for lr_val in lr:
        for wordNgrams_val in wordNgrams:
            for epoch_val in epoch:
                # K-fold CV
                kf = KFold(n_splits=k, shuffle=True)
                # For each fold
                for train_index, test_index in kf.split(YX):
                    YX[train_index].to_csv('train_cv.txt', index = False)
                    # Fit model for this set of parameter values
                    now = datetime.datetime.now()
                    model = ft.FastText.train_supervised('train_cv.txt',
                                                         lr=lr_val,
                                                         wordNgrams=wordNgrams_val,
                                                         epoch=epoch_val)
                    elapsed = (datetime.datetime.now() - now)
                    df_valid=  pd.DataFrame(data=YX[test_index]).dropna()
                    pred = model.predict(df_valid['labels_text'].tolist())
                    pred_filter = list(filter(None, pred[0]))
                    pred_filter = pd.Series(pred_filter).apply(lambda x: re.sub('__label__', '', x[0]))
                    org =df_valid['labels_text'].apply(lambda x: re.sub('__label__', '', x[9:x.find(' ')]))
                    # Compute accuracy for this CV fold
                    fold_accuracy =metrics.accuracy_score(org.values, pred_filter.values)
                    accuracy.append(fold_accuracy)
                    print(fold_accuracy)

                # Compute mean accuracy across 10 folds
                mean_acc = np.mean(accuracy)
                cross_valid_params[mean_acc] = [lr_val, wordNgrams_val, epoch_val]
                print("mean acc" + str(mean_acc))



    max_acc = max(cross_valid_params.keys())
    lr_val, wordNgrams_val, epoch_val = cross_valid_params[max_acc]
    print(cross_valid_params[max_acc])
    precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org.values, pred_filter.values)
    print("Classify Fast Text Precision =", precisions)
    print("Classify Fast Text Recall=", recall)
    print("Classify Fast Text F1 Score =", f1_score)
    accuracy_score = metrics.accuracy_score(org, pred_filter)
    print('Accuracy' + str(accuracy_score))
    train_cross_validated_model(lr_val, wordNgrams_val, epoch_val)




def train_cross_validated_model(lr_val, wordNgrams_val, epoch_val):
    classifier = ft.FastText.train_supervised('train.txt', lr=lr_val, wordNgrams=wordNgrams_val, epoch=epoch_val)
    predictions = classifier.predict(df_test.tweet.tolist())
    if(predictions != None and len(predictions) > 0):
        estimated = pd.Series(predictions[0]).apply(lambda x: re.sub('__label__', '', x[0]))
        org = df_test['labels_text'].apply(lambda x: re.sub('__label__', '', x[9:9+len('labels_text')]))
        precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org, estimated.values)
        print("Classify Fast Text Precision =", precisions)
        print("Classify Fast Text Recall=", recall)
        print("Classify Fast Text F1 Score =", f1_score)

        accuracy_score = metrics.accuracy_score(org, estimated)
        print('Accuracy' + str(accuracy_score))

        num_classes = 8
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray']
        legends = ['anger', 'arousal', 'dominance', 'faith', 'fear', 'joy', 'neutral', 'sadness']
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        count = 0
        for f_score in f1_score:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color=colors[count], alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            count = count + 1

        for i in range(num_classes):
            l, = plt.plot(recall[i], precisions[i], color=colors[i], lw=2)
            lines.append(l)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve of Multi-class Moods for Congress'  ' using FastText')
        plt.legend(lines, legends, loc=1, prop=dict(size=6))
        plt.show()


cross_validate(df_train.labels_text,k = 10,lr = [0.05, 0.1, 0.2],wordNgrams = [1,2,3], epoch = [15,17,20])
#train_cross_validated_model(0.05, 1, 20)
