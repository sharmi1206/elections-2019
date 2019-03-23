from pprint import pprint
import os
import pandas as pd
import matplotlib
import numpy as np
import hashlib
from wordcloud import WordCloud
matplotlib.use('TkAgg')
import matplotlib.pyplot as mplt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]


plot_path = filePath + '/train/sentiments/'
word_disb_path = filePath + '/train/wordstats/1-gram/'

uni_gram_files = ['LokShobaElc2019BJP-freqdist-uni.csv', 'LokShobaElc2019Cong-freqdist-uni.csv', 'LokShobaElc2019Both-freqdist-uni.csv', 'LokShobaElc2019Neutral-freqdist-uni.csv']
bi_gram_files = ['LokShobaElc2019BJP-freqdist-bi.csv', 'LokShobaElc2019Cong-freqdist-bi.csv', 'LokShobaElc2019Both-freqdist-bi.csv', 'LokShobaElc2019Neutral-freqdist-bi.csv']
tri_gram_files = ['LokShobaElc2019BJP-freqdist-tri.csv', 'LokShobaElc2019Cong-freqdist-tri.csv', 'LokShobaElc2019Both-freqdist-tri.csv', 'LokShobaElc2019Neutral-freqdist-tri.csv']
quad_gram_files = ['LokShobaElc2019BJP-freqdist-quad.csv', 'LokShobaElc2019Cong-freqdist-quad.csv', 'LokShobaElc2019Both-freqdist-quad.csv', 'LokShobaElc2019Neutral-freqdist-quad.csv']


plot_path_train = filePath + '/train/sentiments/train-dataset/'
plot_path_test = filePath + '/train/sentiments/test-dataset/'

full_statspath = filePath +'/train/raw/'

files = os.listdir(plot_path)
stats_files = os.listdir(full_statspath)

files = ['.DS_Store', 'LokShobaElc2019BJP-moods.csv', 'LokShobaElc2019Cong-moods.csv', 'LokShobaElc2019Both-moods.csv', 'LokShobaElc2019Neutral-moods.csv']
labels = ['BJP', 'Congress', 'BJP-Congress', 'Neutral']

for i in range(len(stats_files)):
    fname = files[i+1]
    print(fname)

    if(fname.startswith('.')== False and fname.endswith('.csv') == True):
        list_full_data = []
        df_raw =  pd.read_csv(full_statspath + stats_files[i]).dropna()


        sns.set(font_scale=0.8)
        df_BJP =  pd.read_csv(plot_path + files[i+1])
        df_Cong = pd.read_csv(plot_path + files[i+2])
        df_BJP_Cong = pd.read_csv(plot_path + files[i+3])
        fields = ['tweet', 'mood']

        location_df = df_BJP['location'].value_counts()
        filter_loc = location_df[location_df > 35]

        location_df = df_BJP['location'].value_counts()
        filter_loc = location_df[location_df > 35]
        patches, texts, autotexts = mplt.pie(
            filter_loc,
            labels=filter_loc.index.values,
            shadow=False,
            startangle=90,
            pctdistance=0.7, labeldistance=1.15,
            # with the percent listed as a fraction
            autopct='%1.1f%%',
        )
        mplt.axis('equal')
        mplt.tight_layout()
        mplt.show()

        # Create a figure instance, and the two subplots
        fig = mplt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        size = fig.get_size_inches() * fig.dpi  # get fig size in pixels
        ax1.set_title("LokShobha Elections 2019 " + labels[i] + " Sentiments", fontsize = 8.5, loc ='right')
        ax2.set_title("LokShobha Elections 2019 " + labels[i+1] + " Sentiments", fontsize = 8.5, loc ='right')

        # Tell countplot to plot on ax1 with one party and ax2 with another party
        g = sns.countplot(x="mood", data=df_BJP,  palette="PuBuGn_d",  ax=ax1, order = df_BJP['mood'].value_counts().index)
        g = sns.countplot(x="mood", data=df_Cong,  palette="PuBuGn_d",  ax=ax2, order = df_BJP['mood'].value_counts().index)
        mplt.show()

        area = np.pi * 3
        mplt.scatter(df_BJP['compound'],df_BJP['mood'], s=area, alpha=0.5)
        mplt.title('Tweet Intensity for ' + labels[i])
        mplt.xlabel('Sentiment Intensity')
        mplt.ylabel('Moods')
        mplt.show()

        area = np.pi * 3
        mplt.scatter(df_Cong['compound'], df_Cong['mood'], s=area, alpha=0.5)
        mplt.title('Tweet Intensity for ' + labels[i+1])
        mplt.xlabel('Sentiment Intensity')
        mplt.ylabel('Moods')
        mplt.show()

        area = np.pi * 3
        mplt.scatter(df_BJP_Cong['compound'], df_BJP_Cong['mood'], s=area, alpha=0.5)
        mplt.title('Tweet Intensity for ' + labels[i+2])
        mplt.xlabel('Sentiment Intensity')
        mplt.ylabel('Moods')
        mplt.show()

        df = pd.read_csv(plot_path + fname).dropna()
        list_full_data.append(df)
        fields = ['tweet', 'mood']

        sns.set(font_scale=.7)


        combined_df = pd.concat(list_full_data, axis=0, ignore_index=True)
        emoji_df = combined_df['emoji'].value_counts()
        mplt.rcParams['font.size'] = 6.0
        #mplt.title(labels[i], loc='right')

        patches, texts, autotexts = mplt.pie(
            emoji_df,
            labels=emoji_df.index.values,
            shadow=False,
            startangle=90,
            pctdistance=0.7, labeldistance=1.05,
            # with the percent listed as a fraction
            autopct='%1.1f%%',
        )
        mplt.axis('equal')
        mplt.tight_layout()
        mplt.show()

        # Unigram Frequency Distribution
        word_counter_df = pd.read_csv(word_disb_path + uni_gram_files[i])
        word_popular_df = word_counter_df.nlargest(25, columns=['F'])
        word_popular_df['unigram_word'] = word_popular_df.W1
        fig = sns.barplot(x=word_popular_df["unigram_word"], y=word_popular_df["F"])
        sns.set(font_scale=.3)
        mplt.xlabel("Unigram Words", fontsize=10)
        mplt.ylabel("Frequency", fontsize=10)
        mplt.title("LokShobha Elections 2019 " + labels[i], fontsize=10)
        mplt.show(fig)
        # Bigram Frequency Distribution
        sns.set(font_scale=0.5)
        word_disb_path = filePath + '/train/wordstats/2-gram/'
        word_counter_df = pd.read_csv(word_disb_path + bi_gram_files[i])
        word_popular_df['bigram_word'] = word_popular_df.W1 + "  " + word_popular_df.W2
        fig = sns.barplot(x=word_popular_df["bigram_word"], y=word_popular_df["F"])
        sns.set(font_scale=.5)
        mplt.xlabel("Bigram Words", fontsize=10)
        mplt.ylabel("Frequency", fontsize=10)
        mplt.title("LokShobha Elections 2019 " + labels[i], fontsize=10)
        mplt.show(fig)

        df_raw_retweets = df_raw.nlargest(25, columns=['retweet_count'])

        x = df_raw_retweets["full_text"].values
        y = df_raw_retweets["retweet_count"].values

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        fig, ax = mplt.subplots()

        offset = 0.75
        for k in range(len(x)):
            ax.text(offset, k, x[k], color='blue', fontweight='bold', fontsize=7)
            offset = offset + 1

        width = 0.75  # the width of the bars
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, width, color=colors)
        mplt.title(labels[i])
        mplt.xlabel('Retweet Frequency', fontsize=7)
        mplt.ylabel('Tweets', fontsize=7)
        mplt.show()


        df_faith = df[df['mood'] == 'faith']
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_faith.tweet.values))
        mplt.figure(figsize=(12, 10))
        mplt.imshow(wordcloud, interpolation="bilinear")
        mplt.title(labels[i] + "  Faith", fontsize=10)
        mplt.xlabel('Support/Faith')
        mplt.axis("off")
        mplt.show()

        df_fear = df[df['mood'] == 'fear']
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_fear.tweet.values))
        mplt.figure(figsize=(12, 10))
        mplt.imshow(wordcloud, interpolation="bilinear")
        mplt.title(labels[i] + "  Fear", fontsize=10)
        mplt.xlabel('Fear')
        mplt.axis("off")
        mplt.show()

        df_sadness = df[df['mood'] == 'sadness']
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_sadness.tweet.values))
        mplt.figure(figsize=(12, 10))
        mplt.imshow(wordcloud, interpolation="bilinear")
        mplt.xlabel('Sadness')
        mplt.title(labels[i] + "  Sadness", fontsize=10)
        mplt.axis("off")
        mplt.show()

        df_joy = df[df['mood'] == 'joy']
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(str(df_joy.tweet.values))
        mplt.figure(figsize=(12, 10))
        mplt.imshow(wordcloud, interpolation="bilinear")
        mplt.xlabel('Joy')
        mplt.title(labels[i] + "  Joy", fontsize=10)
        mplt.axis("off")
        mplt.show()



