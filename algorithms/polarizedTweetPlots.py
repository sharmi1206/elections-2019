import os
import pandas as pd
import matplotlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
matplotlib.use('TkAgg')
import matplotlib.pyplot as mplt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
import nltk
from nltk.tokenize import *


import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0] + "/train/sentiments/"

files = os.listdir(filePath)


def label_encode(mood):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(mood)
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return integer_encoded

labels = ['BJP', 'Congress', 'BJP-Congress', 'Neutral']
titles = ['Parts of Speech (Adjectives/Adverbs/Proper Nouns) Tagging for BJP', 'Parts of Speech (Adjectives/Adverbs/Proper Nouns) Tagging for Congress', 'Parts of Speech (Adjectives/Adverbs/Proper Nouns) Tagging for BJP and Congress', 'Parts of Speech (Adjectives/Adverbs/Proper Nouns) Tagging for Other Parties']

fcount = 0
for fcount in range(len(files)):
    if(files[fcount].endswith('.csv')):
        combined_df = pd.read_csv(filePath + files[fcount])
        data = combined_df.ix[:, ['tweet', 'mood', 'created_at']]
        data_sentiment = label_encode(combined_df['mood'])
        data_date = pd.to_datetime(combined_df['created_at'])
        n = len(data)

        data['tokenized_tweet'] = data.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)


        for i, row in data.iterrows():
            s = word_tokenize(row[0])
            s = nltk.pos_tag(s)
            data.set_value(i, 'tweet', s)
            data.set_value(i, 'mood', data_sentiment[i])
            data.set_value(i, 'created_at', data_date[i])

        dateMap = {}
        for index, row in data.iterrows():
            adj = 0
            adv = 0
            pn = 0

            for i in range(len(row[0])):
                time = row['created_at']._date_repr
                if row[0][i][1] == 'RB' or row[0][i][1] == 'RBR' or row[0][i][1] == 'RBS':
                    adv = adv + 1
                elif row[0][i][1] == 'JJ' or row[0][i][1] == 'JJR' or row[0][i][1] == 'JJS':
                    adj = adj + 1
                elif row[0][i][1] == 'NNP' or row[0][i][1] == 'NNPS':
                    pn = pn + 1
            if(time in dateMap):
                oldData = dateMap[time]
                dateMap[time] = [adv+oldData[0], adj+oldData[1], pn+oldData[2]]
            else:
                dateMap[time] = [adv, adj, pn]



        sortedDate = sorted(dateMap.items())
        print(sortedDate)

        ndf = pd.DataFrame(0, index=range(len(sortedDate)),
                          columns=['date', 'adv', 'adj', 'nn'])
        count = 0

        for item in sortedDate:
            date_ = item[0]
            vals = item[1]
            ndf.set_value(count, 'date', pd.to_datetime(date_))
            ndf.set_value(count, 'adv', vals[0])
            ndf.set_value(count, 'adj', vals[1])
            ndf.set_value(count, 'nn', vals[2])
            count = count+1

        mplt.plot(ndf['date'], ndf['adv'],'r', label='Adverbs')
        mplt.plot(ndf['date'], ndf['adj'], 'b', label='Adjectives')
        mplt.plot(ndf['date'], ndf['nn'], 'g', label = 'Proper Noun')
        mplt.xlabel('Date')
        mplt.ylabel('Frequency')
        mplt.title(titles[fcount])
        mplt.legend()
        mplt.show()







