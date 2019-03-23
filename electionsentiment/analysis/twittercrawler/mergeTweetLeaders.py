import pandas as pd
import glob
import os
import preprocessor as p
import numpy as np
import re
from analysis.tweetprocessor import preprocessTweets as pt
from analysis.tweetprocessor import processMoods as md

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]

allFiles = []
merge_path = filePath + '/train/raw/'
dirs = os.listdir(merge_path)
merge_path_ld = filePath + '/train/raw/party'
files = os.listdir(merge_path_ld)

allFiles.append(glob.glob(merge_path + "/*.csv"))
allFiles.append(glob.glob(merge_path_ld + "/*.csv"))

list_Modi = []
list_Rahul = []
isBJP = True

target_BJP = ['@narendramodi']
target_Cong = ['@rahulgandhi']

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.MENTION)
clean_BJP = []
clean_Cong = []

for file_ in allFiles:
    for i in range(0, len(file_)):
        df = pd.read_csv(file_[i],index_col=None, header=0).dropna()
        df_BJP = df[df.apply(lambda x: pd.Series(df.full_text).str.contains('@narendramodi'))['full_text']].dropna()
        df_Cong = df[df.apply(lambda x: pd.Series(df.full_text).str.contains('@rahulgandhi'))['full_text']].dropna()


    list_Modi.append(df_BJP)
    list_Rahul.append(df_Cong)


df_BJP = pd.concat(list_Modi, axis = 0, ignore_index = True)
df_BJP = df_BJP.drop_duplicates(subset=['created_at', 'full_text'])
df_BJP = df_BJP[df_BJP.full_text != 'full_text']

df_Cong = pd.concat(list_Rahul, axis = 0, ignore_index = True)
df_Cong = df_Cong.drop_duplicates(subset=['created_at', 'full_text'])
df_Cong = df_Cong[df_Cong.full_text != 'full_text']


df_BJP.to_csv(merge_path + 'Modi.csv', index=False)
df_Cong.to_csv(merge_path + 'Rahul.csv', index=False)

process_files = ["Modi", "Rahul"]

process_path = filePath + "/train/preprocessed/"
for i in range(0, len(process_files)):
    processing_results = pt.preprocess_csv(merge_path + process_files[i] + ".csv", process_path + process_files[i] + ".csv")
    md.processMoods(processing_results, filePath + "/train/sentiments/" + process_files[i] + "-moods.csv")



