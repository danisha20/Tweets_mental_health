# %%
import re
import string
#import numpy as np
import pandas as pd
from time import time

from tqdm.auto import tqdm

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from random import randint

class LabelDepressiveTweets():

    def __init__(self, url):
        self.url_ = url
        self.corpus_, self.df_ = self.importdata()
        self.clean_tweets('description')
        self.dep_words_ = ['overwhelmed','exhausted',
            'distressed','anxiety','anxious',
            'tired','low','depression','depressed',
            'discouraged','desperate','demotivated',
            'insomnia','cry','nervous','worried',
            'lonely','sad','empty']
        self.dep_words_stemmed_ = self.stem_dep_list()
        self.search_tweets()
    
    def importdata(self):
        print('Importing data...')
        st = time()
        col_names = ['label','id','date','query','user','description']
        df = pd.read_csv(self.url_, header=None, encoding = "ISO-8859-1", names = col_names)
        corpus = df.copy()
        end = time()
        print('Finished in {0:.2f} seconds.'.format(end-st))
        return corpus, df

    def process_tweet(self, tweet):
        """Process tweet function.
        Input:
            tweet: a string containing a tweet
        Output:
            tweets_clean: a list of words containing the processed tweet
        """
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                stem_word = stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)

        return tweets_clean

    def clean_tweets(self,col_name):
        print('Cleaning tweets...')
        st = time()
        tqdm.pandas()
        self.df_[col_name] = self.df_.progress_apply(lambda row: self.process_tweet(row[col_name]), axis=1)
        end = time()
        print('Finished in {0:.2f} minutes.'.format((end-st)/60))

        return self

    def stem_dep_list(self):
        ps = PorterStemmer()
        dep_words_stemmed = [ps.stem(word) for word in self.dep_words_]

        return dep_words_stemmed
    
    def search(self, tweet, dep_words_stemmed):
        if len(tweet) > 0:
            for word in tweet:
                if word in dep_words_stemmed:
                    return 1
                else:  
                    return 0
        else:
            return 0

    def search_tweets(self):
        self.df_['flag'] = 0
        self.df_['flag'] = self.df_.progress_apply(lambda row: self.search(row['description'],self.dep_words_stemmed_), axis=1)
        self.df_['flag'] = self.df_['flag'].astype(int)
        print("I've found {} tweets.".format(str(self.df_.flag.sum())))

        return self

    def generate_random(self, avoid):
        rand = int(randint(0, len(self.df_)))
        while True:
            if rand in avoid:
                rand = int(randint(0, len(self.df_)))
            else:
                avoid.append(rand)
                return avoid

    def get_train_data(self):
        df_dep = self.corpus_.loc[self.df_[(self.df_['flag'] == 1)].index, :]
        df_dep['flag'] = 1

        avoid_list_index = self.df_[(self.df_['flag'] == 1)].index.tolist()
        for _ in range(len(df_dep)):
            avoid_list_index = self.generate_random(avoid_list_index)
        non_drep_index = avoid_list_index[12181:]
        non_drep_index.sort()

        df_train = df_dep.append(self.corpus_.loc[non_drep_index,:])
        df_train['flag'] = df_train['flag'].fillna(0)
        df_train['flag'] = df_train.flag.astype(int)

        return df_train

# %%
if __name__ == "__main__":

    data = LabelDepressiveTweets('tweets_database.csv')

    df_train = data.get_train_data()

    df_train.to_csv('train_tweets.csv',index=False)