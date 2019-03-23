import utils
import textblob as tb
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import os
import hashlib
import csv
from sklearn.model_selection import train_test_split
import operator
from nltk import PorterStemmer

retweet_dict = {}
ps = PorterStemmer()

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
ofilePath = fileDir.rsplit('/', 1)[0]
filePath = ofilePath.rsplit('/', 1)[0]



mood_positives =['faith', 'joy']
mood_negatives = ['sadness',  'arousal', 'dominance', 'anger']
sentiments_file_path = filePath +'/publicsentiments/'

POSITIVE_WORDS_FILE = sentiments_file_path + 'positive-words.csv'
NEGATIVE_WORDS_FILE = sentiments_file_path + 'negative-words.csv'
ANGER_WORDS_FILE = sentiments_file_path + 'anger.csv'
HAPPINESS_WORDS_FILE = sentiments_file_path + 'happiness.csv'
JEALOUS_WORDS_FILE = sentiments_file_path + 'jealous.csv'
SHAME_WORDS_FILE = sentiments_file_path + 'shame.csv'
SADNESS_WORDS_FILE = sentiments_file_path + 'sadness.csv'
SUICIDAL_WORDS_FILE = sentiments_file_path + 'suicidal.csv'
FEAR_WORDS_FILE = sentiments_file_path + 'fear.csv'
SURPRISE_WORDS_FILE = sentiments_file_path + 'surprise.csv'


AROUSAL_WORDS_FILE = sentiments_file_path + 'a-scores.csv'
DOMINANCEL_WORDS_FILE = sentiments_file_path + 'd-scores.csv'
AFFECTDM_WORDS_FILE = sentiments_file_path + 'affectdm-scores.csv'
EXCITEMENT_WORDS_FILE = sentiments_file_path + 'excitement.csv'
FAITH_WORDS_FILE  = sentiments_file_path + 'faith.csv'
ALL_WORDS_FILE = sentiments_file_path + 'avd-scores.csv'
SUPPORT_PHRASES = sentiments_file_path + 'support.csv'



positive_words = pd.read_csv(POSITIVE_WORDS_FILE)['Word']
negative_words = pd.read_csv(NEGATIVE_WORDS_FILE)['Word']
all_words = pd.read_csv(ALL_WORDS_FILE)['Word'].str.lower()
v_score = pd.read_csv(ALL_WORDS_FILE)['Valence']
a_score = pd.read_csv(ALL_WORDS_FILE)['Arousal']
d_score = pd.read_csv(ALL_WORDS_FILE)['Dominance']
affect_words  = pd.read_csv(AFFECTDM_WORDS_FILE)['term'].str.lower()
affect_mood = pd.read_csv(AFFECTDM_WORDS_FILE)['AffectDimension'].str.lower().values

shame_words= pd.read_csv(SHAME_WORDS_FILE)['Word'].str.lower()
anger_words = pd.read_csv(ANGER_WORDS_FILE)['Word'].str.lower()
sadness_words = pd.read_csv(SADNESS_WORDS_FILE)['Word'].str.lower()
suicidal_words = pd.read_csv(SUICIDAL_WORDS_FILE)['Word'].str.lower()
happiness_words = pd.read_csv(HAPPINESS_WORDS_FILE)['Word'].str.lower()
jealous_words = pd.read_csv(JEALOUS_WORDS_FILE)['Word'].str.lower()
surprise_words = pd.read_csv(SURPRISE_WORDS_FILE)['Word'].str.lower()
faith_words = pd.read_csv(FAITH_WORDS_FILE)['Word'].str.lower()

support_phrases =  pd.read_csv(SUPPORT_PHRASES)['Word'].str.lower()
#stemmer = SnowballStemmer("english")


def get_n_gram_mood(n_gram_word):

    all_moods = []
    v_count, a_count, d_count, affect_count = 0, 0, 0, 0
    pos_count, neg_count = 0, 0
    sid = SentimentIntensityAnalyzer()
    total_ss = 0
    max_len = 0

    for word in n_gram_word:
        #stem_word = ps.stem(word)
        blob = tb.TextBlob(word)

        context_mood = False
        max_len = max_len + 1
        ss = sid.polarity_scores(word)
        total_ss = ss['compound'] + total_ss

        if len(affect_words[word == affect_words]) > 0:
            affect_count += 1
            itemIndex = (affect_words[word == affect_words]).index[0]
            all_moods.append(affect_mood[itemIndex])
            context_mood = True
        elif len(all_words[word == all_words]) > 0:
            a_count = float(a_score[all_words[word == all_words].index[0]])
            v_count = float(v_score[all_words[word == all_words].index[0]])
            d_count = float(d_score[all_words[word == all_words].index[0]])

        if (len(jealous_words[(jealous_words == word)])):
            all_moods.append('anger')
            context_mood = True
        elif (len(shame_words[(shame_words == word)])):
            all_moods.append('sadness')
            context_mood = True
        elif (len(anger_words[(anger_words == word)])):
            all_moods.append('anger')
            context_mood = True
        elif (len(sadness_words[(sadness_words == word)])):
            all_moods.append('sadness')
            context_mood = True
        elif (len(happiness_words[(happiness_words == word)])):
            all_moods.append('joy')
            context_mood = True
        elif (len(suicidal_words[(suicidal_words == word)])):
            all_moods.append('sadness')
            context_mood = True
        elif (len(suicidal_words[(surprise_words == word)])):
            all_moods.append('joy')
            context_mood = True
        elif (len(faith_words[(faith_words == word)])):
            all_moods.append('faith')
            context_mood = True
        elif (a_count > v_count and a_count > d_count and a_count > 0.50):
            all_moods.append('arousal')
            context_mood = True
        elif (d_count > a_count and d_count > v_count and d_count > 0.50):
            all_moods.append('dominance')
            context_mood = True
        elif (v_count > a_count and v_count > d_count and v_count > 0.50):
             word_polarity = blob.polarity
             if(word_polarity > 0):
                all_moods.append('joy')
             elif(word_polarity < 0):
                 all_moods.append('sadness')
             else:
                 all_moods.append('neutral')
             context_mood = True
        else:
            supportVals = support_phrases.values
            for i in range(len(supportVals)):
                if (supportVals[i] in n_gram_word):
                    all_moods.append('faith')

        if (context_mood == False):
            if word in positive_words.values:
                pos_count += 1
                all_moods.append('joy')
            elif word in negative_words.values:
                neg_count += 1
                all_moods.append('sadness')

    final_ss = float(total_ss/max_len)
    final_mood = determine_mood_combination(all_moods,final_ss, word)
    return final_mood

def determine_priority(mood_list, more_moods, summary):

    final_mood = 'dominance'
    if(len(more_moods) > 0):
        final_mood = more_moods[0]
        return final_mood

    if(summary > 0 ):
        for i in range(0, len(mood_positives)):
            if(mood_positives[i] in mood_list):
                final_mood = mood_positives[i]
                break
    else:
        for i in range(0, len(mood_negatives)):
            if(mood_negatives[i] in mood_list):
                final_mood = mood_negatives[i]
                break

    return final_mood

def calculate_max_mood(mood_list, summary, bDist):
    no_positives = 0
    no_negatives = 0
    mood_freq_dist = {}

    for i in range(len(mood_list)):
        if (mood_list[i] in mood_positives):
            no_positives = no_positives + 1
        elif (mood_list[i] in mood_negatives):
            no_negatives = no_negatives + 1
        if (mood_list[i] in mood_freq_dist):
            mood_freq_dist[mood_list[i]] = mood_freq_dist[mood_list[i]] + 1
        else:
            mood_freq_dist[mood_list[i]] = 0

    max_mood_item = max(mood_freq_dist.items(), key=operator.itemgetter(1))[0]

    if(bDist == True):
        if (no_positives > no_negatives and max_mood_item in mood_negatives):
            if (summary > 0):
                max_mood_item = 'joy'
        elif (no_negatives > no_positives and max_mood_item in mood_positives):
            if (summary < 0):
                max_mood_item = 'sadness'
    return max_mood_item


def determine_mood_combination(mood_list, summary, full_word):
    max_mood_item = ''

    if (len(mood_list) == 0):
        if(summary > 0):
            max_mood_item = 'joy'
        elif(summary < 0):
            max_mood_item = 'sadness'
        else:
            #print(full_word, mood_list, summary)
            max_mood_item = 'neutral'
    else:
        max_mood_item = calculate_max_mood(mood_list, summary, True)

    return max_mood_item


def search_context_word(word, tweet):
    all_moods = []
    affect_count = 0
    v_count, a_count, d_count = 0, 0, 0
    context_mood = False

    if len(affect_words[word == affect_words]) > 0:
        affect_count += 1
        # all_moods.append(affect_mood[affect_words[word == affect_words].index[0]])
        itemIndex = (affect_words[word == affect_words]).index[0]
        all_moods.append(affect_mood[itemIndex])
        context_mood = True
    elif len(all_words[word == all_words]) > 0:
        a_count = float(a_score[all_words[word == all_words].index[0]])
        v_count = float(v_score[all_words[word == all_words].index[0]])
        d_count = float(d_score[all_words[word == all_words].index[0]])

    if (len(jealous_words[(jealous_words == word)])):
        all_moods.append('anger')
        context_mood = True
    elif (len(shame_words[(shame_words == word)])):
        all_moods.append('sadness')
        context_mood = True
    elif (len(anger_words[(anger_words == word)])):
        all_moods.append('anger')
        context_mood = True
    elif (len(sadness_words[(sadness_words == word)])):
        all_moods.append('sadness')
        context_mood = True
    elif (len(happiness_words[(happiness_words == word)])):
        all_moods.append('joy')
        context_mood = True
    elif (len(suicidal_words[(suicidal_words == word)])):
        all_moods.append('sadness')
        context_mood = True
    elif (len(suicidal_words[(surprise_words == word)])):
        all_moods.append('joy')
        context_mood = True
    elif (len(faith_words[(faith_words == word)])):
        all_moods.append('faith')
        context_mood = True
    elif (a_count > v_count and a_count > d_count and a_count > 0.50):
        all_moods.append('arousal')
        context_mood = True
    elif (d_count > a_count and d_count > v_count and d_count > 0.50):
        all_moods.append('dominance')
        context_mood = True
        # elif (v_count > a_count and v_count > d_count and v_count > 0.50):
        # all_moods.append('joy')
    else:
        supportVals = support_phrases.values
        for i in range(len(supportVals)):
            if (supportVals[i] in tweet):
                all_moods.append('faith')

    return all_moods, context_mood, a_count, v_count, d_count, affect_count



def determine_vocab_mood(word, tweet, ss):
    # word = ps.stem(full_word)
    word_mood = []
    pos_count, neg_count = 0, 0
    word_mood, context_mood, a_count, v_count, d_count, affect_count = search_context_word(word,tweet)

    if (context_mood == False):
        word_mood, context_mood, a_count, v_count, d_count, affect_count = search_context_word(ps.stem(word), tweet)
        if (context_mood == False):
            if word in positive_words.values:
                pos_count += 1
                word_mood.append('joy')
            elif word in negative_words.values:
                neg_count += 1
                word_mood.append('sadness')


    return word_mood, pos_count, neg_count


def classifyLabelMoods(processing_results):

    predictions =[]
    for i in range(0, len(processing_results)):
        all_moods = []
        pos_count, neg_count = 0, 0
        [tweet_id, tweet, creation_date, favourites_count, statuses_count, followers_count, retweeted, retweet_count,
         processed_retweet, location, hashtags, user_mentions, symbols, urls, emoji] = processing_results[i]

        tweetHash = (hashlib.md5(tweet.encode('utf-8'))).hexdigest()
        if (tweetHash not in retweet_dict.keys()):
            retweet_dict[tweetHash] = retweet_count
            try:
                tweetIdType = type(int(tweet_id))
            except ValueError:
                tweet_id = "1234"
                continue

            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(tweet)
            procList = []
            procList.append(ss['neg'])
            procList.append(ss['neu'])
            procList.append(ss['pos'])
            procList.append(ss['compound'])

            for word in tweet.split():
                word_mood, pos_count, neg_count = determine_vocab_mood(word, tweet, ss)
                if(len(word_mood) > 0):
                    all_moods.append(word_mood[0])

            if (emoji != ''):
                all_moods.append(emoji)

            final_mood = determine_mood_combination(all_moods, ss['compound'], word)
            print(final_mood, emoji, all_moods, tweet)


            blob = tb.TextBlob(tweet)

            if pos_count >= neg_count:
                prediction = 1
            elif blob.sentiment.polarity < 0:
                prediction = -1
            else:
                prediction = 0

            predictions.append((tweet_id, prediction, "{0:.2f}".format(blob.sentiment.polarity),
                                    "{0:.2f}".format(blob.sentiment.subjectivity), ss['compound'], ss['neg'], ss['neu'],
                                    ss['pos'], final_mood, tweet, creation_date, favourites_count, statuses_count, followers_count, retweeted, retweet_count,
                                processed_retweet, location, hashtags, user_mentions, symbols, urls, emoji))

    return predictions


def dump_to_csv(results, csv_file):
    moodCsvWriter = csv.writer(open(csv_file, 'w'), delimiter=',')
    moodCsvWriter.writerow(['id', 'prediction', 'polarity', 'subjectivity', 'compound', 'neg', 'neu', 'pos', 'mood', 'tweet', 'created_at', 'favourites_count', 'statuses_count', 'followers_count', 'retweeted', 'retweet_count', 'retweeted_text', 'location', 'hashtags', 'user_mentions', 'symbols', 'urls', 'emoji'])
    for i in range(0, len(results)):
        moodCsvWriter.writerow(list(results[i]))

    lastInd = csv_file.rfind('/')
    train_file_name = csv_file[0: lastInd] + "/train-dataset/" + csv_file[lastInd+1:len(csv_file)-4] + "-train.csv"
    test_file_name = csv_file[0: lastInd] + "/test-dataset/" +  csv_file[lastInd+1:len(csv_file) - 4] + "-test.csv"

    dataset = pd.read_csv(filePath +'/train/sentiments/LokShobaElc2019BJP-moods.csv')

    train, test = train_test_split(dataset[:-1], test_size=0.2)

    train.to_csv(train_file_name, index=False, index_label=False, columns=['id', 'prediction', 'polarity', 'subjectivity', 'compound', 'neg', 'neu', 'pos', 'mood', 'tweet', 'created_at', 'favourites_count', 'statuses_count', 'followers_count', 'retweeted', 'retweet_count', 'retweeted_text', 'location', 'hashtags', 'user_mentions', 'symbols', 'urls', 'emoji'])
    test.to_csv(test_file_name, index=False, index_label=False, columns=['id', 'prediction', 'polarity', 'subjectivity', 'compound', 'neg', 'neu', 'pos', 'mood', 'tweet', 'created_at', 'favourites_count', 'statuses_count', 'followers_count', 'retweeted', 'retweet_count', 'retweeted_text', 'location', 'hashtags', 'user_mentions', 'symbols', 'urls', 'emoji'])


def processMoods(processing_results, mood_file_name):
    print(mood_file_name, len(processing_results))
    predictions = classifyLabelMoods(processing_results)
    dump_to_csv(predictions, mood_file_name)

