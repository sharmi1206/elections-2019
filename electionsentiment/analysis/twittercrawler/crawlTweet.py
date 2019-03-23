import tweepy
import sys
import hashlib
import csv
import regex as re
import json
import preprocessor as p

#fb app_id 2099605227036254
#app_secret : 4cd29e87cf9feea872b20a67b60daba5
#https://www.kaggle.com/erikbruin/text-mining-the-clinton-and-trump-election-tweets

API_KEY = "XZG9Aa26fYdKQSaTjyboOb9wu"
API_SECRET = "cEWKMDDdFqluZG8t3OibbHXL0ZXif8cW4RLgH1sLDCl5D6W7AE"

# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)


#############################################################

search_congress = ['congress', 'gandhi', 'sonia']
search_bjp= ['modi', 'bjp', 'narendramodi']
search_both= ['congress', 'bjp']
search_anti =  ['rahulvsmodi', 'congressvsbjp']
search_countries = ['maldives', 'japan', 'africa', 'usa', 'canada', 'australia', 'vancouver', 'nigeria', 'philipines', 'finland', 'côte_d’ivoire']

retweet_dict = {}

# Open/Create a file to append data
csvFile_CO = open('../../train/raw/LokShobaElc2019Cong.csv', 'a')
csvFile_BJ = open('../../train/raw/LokShobaElc2019BJP.csv', 'a')
csvFile_BO = open('../../train/raw/LokShobaElc2019Both.csv', 'a')
csvFile_NEU = open('../../train/raw/LokShobaElc2019Neutral.csv', 'a')


#csv Writer
csvCongWriter = csv.writer(csvFile_CO)
csvBjpWriterBj = csv.writer(csvFile_BJ)
csvBothWriterBo = csv.writer(csvFile_BO)
csvWriterNeutral = csv.writer(csvFile_NEU)
#####################################

hastags = ['#congresswins']

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# Continue with rest of analysis


import sys
import jsonpickle
import os

searchQuery = ['#ResultsWithNDTV']  # this is what we're searching for

"""
'#congresswins', '#bjpwins', '#CongressVsBJP',
'#RahulVsModi', '#LokSabha' '#LokSabhaElections',
'#LokSabhaElections2019', 'ModiVsGrandAlliance', '#Elections2019',
'#2019LokSabha', '#BJPIndia2019' '#Congress2019', '#BJP2019', '#ModiForPM2019'
'#GeneralElections' '#GeneralElections2019', '#ModiForPM2019', '#RahulForPM2019', '#ResultsWithNDTV'
"""
maxTweets = 100000000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
fName = 'tweets.txt' # We'll store the tweets in a text file.


# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = "2014-01-01"

# If result only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))


def writeToCSVResults(jsonData, result_both, result_anti, result_cong, result_bjp, clean_text, clean_retweet_text):
    retweeted_text = clean_retweet_text
    if('location' in jsonData['user'] and len(jsonData['user']['location']) > 0):
        location = jsonData['user']['location'].replace(',', '')
    else:
        location = 'Unknown'

    if (result_both or result_anti):


        csvBothWriterBo.writerow([jsonData['id_str'], clean_text,
                                 jsonData['created_at'], jsonData['user']['favourites_count'],
                                 jsonData['user']['statuses_count'],
                                 jsonData['user']['followers_count'],
                                 jsonData['retweeted'], jsonData['retweet_count'],
                                 retweeted_text,
                                  location, len(jsonData['entities']['hashtags']), len(jsonData['entities']['user_mentions']), len(jsonData['entities']['symbols']), len(jsonData['entities']['urls'])])

    elif (result_cong):

        csvCongWriter.writerow([jsonData['id_str'], clean_text,
                                 jsonData['created_at'], jsonData['user']['favourites_count'],
                                 jsonData['user']['statuses_count'],
                                 jsonData['user']['followers_count'],
                                 jsonData['retweeted'], jsonData['retweet_count'],
                                 retweeted_text,
                                 location, len(jsonData['entities']['hashtags']), len(jsonData['entities']['user_mentions']), len(jsonData['entities']['symbols']), len(jsonData['entities']['urls'])])

    elif (result_bjp):

        csvBjpWriterBj.writerow([jsonData['id_str'],clean_text,
                                 jsonData['created_at'], jsonData['user']['favourites_count'],
                                 jsonData['user']['statuses_count'],
                                 jsonData['user']['followers_count'],
                                 jsonData['retweeted'], jsonData['retweet_count'],
                                 retweeted_text,
                                 location, len(jsonData['entities']['hashtags']), len(jsonData['entities']['user_mentions']), len(jsonData['entities']['symbols']), len(jsonData['entities']['urls'])])


    else:
        csvWriterNeutral.writerow([jsonData['id_str'], clean_text,
                                   jsonData['created_at'], jsonData['user']['favourites_count'],
                                   jsonData['user']['statuses_count'],
                                   jsonData['user']['followers_count'],
                                   jsonData['retweeted'], jsonData['retweet_count'],
                                   retweeted_text,
                                   location, len(jsonData['entities']['hashtags']), len(jsonData['entities']['user_mentions']), len(jsonData['entities']['symbols']), len(jsonData['entities']['urls'])])

x = 0
with open(fName, 'w') as f:
    for i in range(0, len(searchQuery)):
        print("Downloading hastag" + searchQuery[i])
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery[i], count=tweetsPerQry,lang="en",tweet_mode='extended')
                    else:
                        new_tweets = api.search(q=searchQuery[i], count=tweetsPerQry,lang="en",tweet_mode='extended',
                                                 since='2018-02-08', until='2019-02-25')
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery[i], count=tweetsPerQry,lang="en",tweet_mode='extended',
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery[i], count=tweetsPerQry,lang="en",tweet_mode='extended',
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    p.set_options(p.OPT.URL,p.OPT.HASHTAG)
                    clean_text = p.clean(tweet.full_text)
                    print(tweet.full_text)



                    print("Clean Full text " + clean_text + "           " + tweet.full_text + str(x))
                    result_cnties = any([re.search(w, tweet.full_text.lower()) for w in search_countries])
                    result_cong = any([re.search(w, tweet.full_text.lower()) for w in search_congress])
                    result_bjp = any([re.search(w, tweet.full_text.lower()) for w in search_bjp])
                    result_both = all([re.search(w, tweet.full_text.lower()) for w in search_both])
                    result_anti = all([re.search(w, tweet.full_text.lower()) for w in search_anti])
                    if(result_cnties == False):

                        tweetHash = (hashlib.md5(tweet.full_text.encode('utf-8'))).hexdigest()
                        jsonData = json.loads(jsonpickle.encode(tweet._json, unpicklable=False))


                        if (tweetHash not in retweet_dict.keys()):

                            f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                                    '\n')
                            clean_retweet_text = ''

                            if ('retweeted_status' in jsonData):
                                retweeted_text = jsonData['retweeted_status']['full_text']
                                clean_retweet_text = p.clean(retweeted_text)

                            writeToCSVResults(jsonData, result_both, result_anti, result_cong, result_bjp, clean_text, clean_retweet_text)
                            retweet_dict[tweetHash] = tweet.retweet_count


                            tweetCount += len(new_tweets)
                            print("tweet count" + str(tweetCount))
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

