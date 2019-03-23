import pandas as pd
import glob
import os

import inspect
fileDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
filePath = fileDir.rsplit('/', 1)[0]

path =filePath + '/old-data/'
dirs = os.listdir(path)
allFiles = []
merge_path = filePath + '/train/raw/LokShobaElc2019'

for i in range(len(dirs)):
    allFiles.append(glob.glob(path + dirs[i] + "/*.csv"))

list_BJP = []
list_Cong = []
list_Both = []
list_Neutral = []

for file_ in allFiles:
    for i in range(0, len(file_)):
        if('BJP' in file_[i]):
            df_BJP = pd.read_csv(file_[i],index_col=None, header=0)
        if ('Cong' in file_[i]):
            df_Cong = pd.read_csv(file_[i], index_col=None, header=0)
        if ('Both' in file_[i]):
            df_Both = pd.read_csv(file_[i], index_col=None, header=0)
        if ('Neutral' in file_[i]):
            df_Neutral = pd.read_csv(file_[i], index_col=None, header=0)

    list_BJP.append(df_BJP)
    list_Cong.append(df_Cong)
    list_Both.append(df_Both)
    list_Neutral.append(df_Neutral)

df_BJP = pd.concat(list_BJP, axis = 0, ignore_index = True)
df_BJP = df_BJP.drop_duplicates(subset=['created_at', 'full_text'])
df_BJP = df_BJP[df_BJP.full_text != 'full_text']

df_Cong = pd.concat(list_Cong, axis = 0, ignore_index = True)
df_Cong = df_Cong.drop_duplicates(subset=['created_at', 'full_text'])
df_Cong = df_Cong[df_Cong.full_text != 'full_text']

df_Both = pd.concat(list_Both, axis = 0, ignore_index = True)
df_Both = df_Both.drop_duplicates(subset=['created_at', 'full_text'])
df_Both = df_Both[df_Both.full_text != 'full_text']


df_Neutral = pd.concat(list_Neutral, axis = 0, ignore_index = True)
df_Neutral = df_Neutral.drop_duplicates(subset=['created_at', 'full_text'])
df_Neutral = df_Neutral[df_Neutral.full_text != 'full_text']


df_BJP.to_csv(merge_path + 'BJP.csv', index=False)
df_Cong.to_csv(merge_path + 'Cong.csv', index=False)
df_Both.to_csv(merge_path + 'Both.csv', index=False)
df_Neutral.to_csv(merge_path + 'Neutral.csv', index=False)

