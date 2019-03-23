To run the program

1. Crawl tweets :

cd  electionsentiment/analysis/twittercrawler

python crawlTweet.py


2. Merge tweeets crawled weekly [ electionsentiment/old-data contains tweets crawled in past weeks, that will be merged]

python mergeTweets.py

The above command will generate the merged files corresponding to Bjp, Congress, both parties , other parties at

electionsentiment/train/raw/LokShobaElc2019... .csv


3. Clean Tweets, process hashtags, url , emojis and generate train and test datasets

cd   electionsentiment/analysis/tweetprocessor

python preprocessTweets.py


4. Run Analytics for party wise comparison charts and polarized tweet classification

cd electionsentiment/algorithms

python sentimentPlots.py

python polarizedTweetPlots.py


5. Run mood prediction algorithms

a.  cd electionsentiment/algorithms

b. python fasttextClassify.py

c. python nltkClassify.py

d. python bagging.py

e. python boosting.py

f. python stacking.py

g. python  classifygloveattlstm.py [Deep learning models based on Word2Vec Embeddings]

10 python classifyw2veclstm.py  [Deep learning LSTM based models]

11 python classifyw2veccnn.py  [Deep learning CNN based models]