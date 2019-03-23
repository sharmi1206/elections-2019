import pandas as pd

def processStopWords():
    df = pd.read_csv('../../publicsentiments/stopwords.csv')
    stop_words = set(df['Word'])
    return stop_words

if __name__ == '__main__':
    processStopWords()
