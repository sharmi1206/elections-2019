import re
import sys
import pandas as pd
import os
from collections import Counter
import pickle
import emoji
from analysis.tweetprocessor import processStopWords as pw
import traceback
from nltk import PorterStemmer

stop_words = pw.processStopWords()
ps = PorterStemmer()

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
ofilePath = fileDir.rsplit('/', 1)[0]
filePath = ofilePath.rsplit('/', 1)[0]



csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.append(csfp)

path = filePath + '/Emojis/'

dirEmojiUni = path + "Unicode"
dirEmojiTxt = path + "Text_Based"

from analysis.tweetprocessor import processMoods as md
from analysis.tweetprocessor import processEmojis as pe

unicodeEmojiList = pe.readUnicode(dirEmojiUni)
emoticonList = pe.readEmoticon(dirEmojiTxt)
all_emoji = emoji.UNICODE_EMOJI.keys()
all_emoji_vals = emoji.UNICODE_EMOJI.values()

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;..')
    # Convert more than 2 letter repetitions to 2 letter
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
   # # Remove 2 or more dots
    #word = re.sub(r'(..)', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._][^0-9]*$', word) is not None)


def handle_emojis(tweet):
    fullStr = ''
    mood = ''
    out = re.findall(r'[^\w\s,]', tweet)
    for i in range(len(out)):
        fullStr = (out[i].encode('unicode-escape').upper()).decode('utf8')[1:]
        if (fullStr in unicodeEmojiList):
            #print("match found" + str(unicodeEmojiList[fullStr]))
            mood = str(unicodeEmojiList[fullStr][0])


    if('joy' in mood):
        mood = 'joy'
    elif('angry' in mood):
        mood = 'angry'
    elif('sad' in mood):
        mood = 'sad'
    elif ('sad' in mood):
        mood = 'sad'
    elif('neutral' in mood):
        mood = 'neutral'
    elif ('arousal' in mood):
        mood = 'arousal'
    elif ('dominance' in mood):
        mood = 'dominance'
    elif ('surprise' in mood):
        mood = 'arousal'


    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return mood


def preprocess_tweet(tweet):
    processed_tweet = []
    emoji = []
    # Convert to lower case
    if(tweet != None):
        tweet = tweet.lower()
        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', '', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Replace ,  with space
        tweet = re.sub(r'\,', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')
        # Replace emojis with either EMO_POS or EMO_NEG
        emoji = handle_emojis(tweet)
        #if(emoji != ''):
            #print("Mood is " + emoji)
        # Replace multiple spaces with a single space
        tweet = re.sub(r'\s+', ' ', tweet)
        words = tweet.split()

        for word in words:
            word = preprocess_word(word)
            if is_valid_word(word) and word not in stop_words:
                if ('b' not in word):
                    processed_tweet.append(word)
                    #word = str(porter_stemmer.stem(word))

        return ' '.join(processed_tweet) , emoji

def get_ngram_freqdist(ngrams):
    freq_dict = {}
    for ngram in ngrams:
        if(ngram in freq_dict):
            freq_dict[ngram] += 1
        else:
            freq_dict[ngram] = 1
    counter = Counter(freq_dict)
    return counter


def get_ngrams(tweet_words, n):
    ngrams = []
    num_words = len(tweet_words)
    for i in range(num_words -(n-1)):
        lookUpTweets = []

        for j in range(i, i+n):
            lookUpTweets.append(tweet_words[j])

        ngrams.append(tuple(lookUpTweets))

    return ngrams


def get_ngrams_all(tweet_words, n):
    ngrams = []
    num_words = len(tweet_words)
    for i in range(num_words -(n-1)):
        ngrams.append((tweet_words[i], tweet_words[i + 1], tweet_words[i + 2], tweet_words[i + 3]))
    return ngrams

def get_bigrams(tweet_words):
    bigrams = []
    num_words = len(tweet_words)
    for i in range(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams

def preprocess_csv(csv_file_name, processed_file_name):
    emoji = []
    save_to_file = open(processed_file_name, 'w')

    df = pd.read_csv(csv_file_name)
    tweetText = df['full_text']
    retweetText = df['retweeted_text']
    tweet_id = df['id_str']
    save_to_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('id_str', 'full_text', 'created_at', 'favourites_count', 'statuses_count', 'followers_count', 'retweeted', 'retweet_count', 'retweeted_text', 'location', 'hashtags', 'user_mentions', 'symbols', 'urls'))
    processed_tweet = ''
    processed_retweet = ''
    results = []

    for i in range(0, len(tweetText)):

        if (pd.isnull(tweetText[i]) == False):
            processed_tweet, emoji = preprocess_tweet(tweetText[i])

        if(pd.isnull(retweetText[i]) == False):
            processed_retweet, emoji = preprocess_tweet(retweetText[i])

        save_to_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (tweet_id[i], processed_tweet,df['created_at'][i],df['favourites_count'][i],df['statuses_count'][i],df['followers_count'][i],df['retweeted'][i],df['retweet_count'][i],processed_retweet,df['location'][i], df['hashtags'][i], df['user_mentions'][i], df['symbols'][i], df['urls'][i]))
        results.append([tweet_id[i], processed_tweet,df['created_at'][i],df['favourites_count'][i],df['statuses_count'][i],df['followers_count'][i],df['retweeted'][i],df['retweet_count'][i],processed_retweet,df['location'][i], df['hashtags'][i], df['user_mentions'][i], df['symbols'][i], df['urls'][i], emoji])

    save_to_file.close()
    print ('\nSaved processed tweets to: %s' % processed_file_name)
    return results

def evaluate_mood_n_grams():
    return

def analyze_tweet(tweet, user_mentions, urls, hash_tags):
    result = {}
    result['MENTIONS'] = user_mentions
    result['URLS'] = urls
    result['HASHTAGS'] = hash_tags
    words = tweet.split()

    result['WORDS'] = len(words)
    bigrams = get_ngrams(words,2)
    result['BIGRAMS'] = len(bigrams)

    trigrams = get_ngrams(words, 3)
    result['TRIGRAMS'] = len(trigrams)

    quadgrams = get_ngrams(words,4)
    result['QUADGRAMS'] = len(quadgrams)

    return result, words, bigrams, trigrams, quadgrams


def getTweetStats(preprocessed_file_name, ipkl_file_name):

    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    num_mentions, max_mentions, max_hashtags = 0, 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls , num_hashtags = 0, 0, 0
    num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
    num_bigrams, num_unique_bigrams = 0, 0
    num_trigrams, num_unique_trigrams = 0, 0
    num_quadgrams, num_unique_quadgrams = 0, 0

    all_words = []
    all_bigrams = []
    all_trigrams = []
    all_quadgrams = []
    all_tri_moods = []
    all_fi_moods = []

    with open(preprocessed_file_name, 'r') as csv:
        lines = csv.readlines()
        for i, line in enumerate(lines):
            tweet = ''
            colCount = len(line.strip().split(','))
            if(colCount == 14):
                t_id, tweet, created_at, favourites_count, statuses_count, followers_count, retweeted, retweet_count, retweeted_text, location, hastags, user_mentions, symbols, urls = line.strip().split(',')
            if(i > 0):
                try:
                    result, words, bigrams, trigrams, quadgrams = analyze_tweet(tweet, user_mentions, urls, hastags)
                    if(result['MENTIONS'] != 'user_mentions'):
                        num_mentions += int(float(result['MENTIONS']))
                    else:
                        num_mentions = 0

                    max_mentions = max(max_mentions, num_mentions)
                    num_urls += int(float(result['URLS']))
                    max_urls = max(max_urls, int(float(result['URLS'])))
                    num_hashtags += int(float(result['HASHTAGS']))
                    max_hashtags = max(max_hashtags, int(float(result['HASHTAGS'])))
                    num_words += int(float(result['WORDS']))
                    min_words = min(min_words, int(float(result['WORDS'])))
                    max_words = max(max_words, int(float(result['WORDS'])))
                    all_words.extend(words)
                    num_bigrams += result['BIGRAMS']
                    num_trigrams += result['TRIGRAMS']
                    num_quadgrams += result['QUADGRAMS']
                    all_bigrams.extend(bigrams)
                    all_trigrams.extend(trigrams)
                    all_quadgrams.extend(quadgrams)
                    num_tweets = num_tweets + 1

                except Exception as e:
                    print("type error: " + str(e))
                    print(traceback.format_exc())

        unique_words = list(set(all_words))
        with open(ipkl_file_name + '-unique.txt', 'w') as uwf:
            uwf.write('\n'.join(unique_words))
        num_unique_words = len(unique_words)
        num_unique_bigrams = len(set(all_bigrams))
        num_unique_trigrams = len(set(all_trigrams))
        num_unique_quadgrams = len(set(all_quadgrams))

        print('\nCalculating frequency distribution')

        # Unigrams
        uni_freq_dist = get_ngram_freqdist(all_words)
        pkl_file_name = ipkl_file_name + '-freqdist.pkl'
        lind = pkl_file_name.rfind('wordstats/')
        maxlen = lind+len('wordstats/')
        pkl_file_name_wdir = pkl_file_name[:maxlen] + "1-gram/" + pkl_file_name[maxlen : len(pkl_file_name)]

        uni_csv_file_name = ipkl_file_name + '-freqdist-uni.csv'
        csv_file_name_wdir = uni_csv_file_name[:maxlen] + "1-gram/" + uni_csv_file_name[maxlen: len(uni_csv_file_name)]

        columns = ['W1', 'F']
        df = pd.DataFrame(columns=columns)
        cnt = 0
        for unigram in uni_freq_dist:
            uList = []
            key = unigram
            uvalue = uni_freq_dist[key]
            uList.append(key)
            uList.append(uvalue)
            df.loc[cnt] = uList
            cnt = cnt + 1
        sorteddf = df.sort_values(by=['F'], ascending=False)
        sorteddf.to_csv(csv_file_name_wdir, index=False)

        # Bigrams
        bigram_freq_dist = get_ngram_freqdist(all_bigrams)
        bi_pkl_file_name = ipkl_file_name + '-freqdist-bi.pkl'
        lind = bi_pkl_file_name.rfind('wordstats/')
        maxlen = lind + len('wordstats/')
        pkl_file_name_wdir = bi_pkl_file_name[:maxlen] + "2-gram/" + bi_pkl_file_name[maxlen: len(bi_pkl_file_name)]
        bi_csv_file_name = ipkl_file_name + '-freqdist-bi.csv'
        csv_file_name_wdir = bi_csv_file_name[:maxlen] + "2-gram/" + bi_csv_file_name[maxlen: len(bi_csv_file_name)]


        columns = ['W1', 'W2', 'F']
        df = pd.DataFrame(columns=columns)
        cnt = 0
        for bgram in bigram_freq_dist:
            bList = []
            key = bgram
            bvalue = bigram_freq_dist[key]
            bList =list(key)
            bList.append(bvalue)
            df.loc[cnt] = bList
            cnt = cnt +1
        sorteddf = df.sort_values(by = ['F'], ascending=False)
        sorteddf.to_csv(csv_file_name_wdir, index = False)

        # Trigrams
        tri_pkl_file_name = ipkl_file_name + '-freqdist-tri.pkl'
        trigram_freq_dist = get_ngram_freqdist(all_trigrams)
        lind = tri_pkl_file_name.rfind('wordstats/')
        maxlen = lind + len('wordstats/')
        pkl_file_name_wdir = tri_pkl_file_name[:maxlen] + "3-gram/" + tri_pkl_file_name[maxlen: len(tri_pkl_file_name)]

        tri_csv_file_name = ipkl_file_name + '-freqdist-tri.csv'
        csv_file_name_wdir = tri_csv_file_name[:maxlen] + "3-gram/" + tri_csv_file_name[maxlen: len(tri_csv_file_name)]

        columns = ['W1', 'W2', 'W3','F']
        df = pd.DataFrame(columns=columns)
        cnt = 0

        try:
            for trigram in trigram_freq_dist:
                triList = []
                key = trigram
                trivalue = trigram_freq_dist[key]
                triList = list(key)
                triList.append(trivalue)
                df.loc[cnt] = triList
                cnt = cnt + 1

        except Exception as e:
            print(traceback.format_exc())

        sorteddf = df.sort_values(by=['F'], ascending=False)
        sorteddf.to_csv(csv_file_name_wdir, index=False)

        fi_csv_file_name = ipkl_file_name + '-freqdist-quad.csv'
        csv_file_name_wdir = fi_csv_file_name[:maxlen] + "4-gram/" + fi_csv_file_name[maxlen: len(fi_csv_file_name)]

        quad_pkl_file_name = ipkl_file_name + '-freqdist-quad.pkl'
        quadgram_freq_dist = get_ngram_freqdist(all_quadgrams)
        lind = quad_pkl_file_name.rfind('wordstats/')
        maxlen = lind + len('wordstats/')
        pkl_file_name_wdir = quad_pkl_file_name[:maxlen] + "4-gram/" + quad_pkl_file_name[maxlen: len(quad_pkl_file_name)]

        columns = ['W1', 'W2', 'W3', 'W4' ,'F']
        df = pd.DataFrame(columns=columns)
        cnt = 0
        for figram in quadgram_freq_dist:

            key = figram
            fivalue = quadgram_freq_dist[key]
            fiList = list(key)
            fiList.append(fivalue)
            df.loc[cnt] = fiList
            cnt = cnt + 1

        sorteddf = df.sort_values(by=['F'], ascending=False)
        sorteddf.to_csv(csv_file_name_wdir, index=False)


        print('Saved bi-frequency distribution to %s' % bi_pkl_file_name)
        print('Saved tri-frequency distribution to %s' % tri_pkl_file_name)
        print('\n[Analysis Statistics]')
        print('Tweets => Total: %d, Positive: %d, Negative: %d' % (num_tweets, num_pos_tweets, num_neg_tweets))
        print('User Mentions => Total: %d, Avg: %.4f, Max: %d' % (num_mentions, num_mentions / float(num_tweets), max_mentions))
        print('URLs => Total: %d, Avg: %.4f, Max: %d' % (num_urls, num_urls / float(num_tweets), max_urls))
        print('HASHTAGS => Total: %d, Avg: %.4f, Max: %d' % (num_hashtags, num_hashtags / float(num_tweets), max_hashtags))

        #print('Emojis => Total: %d, Positive: %d, Negative: %d, Avg: %.4f, Max: %d' % (num_emojis, num_pos_emojis, num_neg_emojis, num_emojis / float(num_tweets), max_emojis))
        print('Words => Total: %d, Unique: %d, Avg: %.4f, Max: %d, Min: %d' % (num_words, num_unique_words, num_words / float(num_tweets), max_words, min_words))
        print('Bigrams => Total: %d, Unique: %d, Avg: %.4f' % (num_bigrams, num_unique_bigrams, num_bigrams / float(num_tweets)))
        print('Trigrams => Total: %d, Unique: %d, Avg: %.4f' % (num_trigrams, num_unique_trigrams, num_trigrams / float(num_tweets)))
        print('Quadgrams => Total: %d, Unique: %d, Avg: %.4f' % (num_quadgrams, num_unique_quadgrams, num_quadgrams / float(num_tweets)))


if __name__ == '__main__':

    path = filePath + '/train/'
    csv_file_names = os.listdir(path + "raw/")


    for i in range(0, len(csv_file_names)):
        if('csv' in csv_file_names[i]):
            preprocessed_file_name =  path  + "preprocessed/" + csv_file_names[i][:-4] + ".csv"
            mood_file_name =  path  + "sentiments/" + csv_file_names[i][:-4] + "-moods.csv"
            processing_results  = preprocess_csv(path + "raw/" + csv_file_names[i], preprocessed_file_name)
            pkl_file_name = path + "wordstats/" + csv_file_names[i][:-4]
            getTweetStats(preprocessed_file_name, pkl_file_name)
            md.processMoods(processing_results, mood_file_name)




