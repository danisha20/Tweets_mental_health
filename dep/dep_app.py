#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import logging, io, base64, os, datetime, sys
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib

import requests
import urllib.parse

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt            # library for visualization
import random                              # pseudo-random number generator

from wordcloud import WordCloud
# Preprocessing
import nltk # Python library for NLP
import matplotlib.pyplot as plt            # library for visualization
import random                              # pseudo-random number generator
import re                                  # library for regular expression operations
import string                              # for string operations
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer, SnowballStemmer    # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn import feature_extraction
from collections import Counter
from nltk.util import ngrams

import re
from sklearn import feature_extraction
import nltk
from string import punctuation
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


class CleanTweet():

    def __init__(self,  stemm = False, lemm =True):
        #self.tweet_ = tweet
        self.stemm_ = stemm
        self.lemm_ = lemm
        self.stemmer_ = nltk.stem.SnowballStemmer('english')
        self.lemmatizer_ = nltk.stem.WordNetLemmatizer()
        self.vect_wc = feature_extraction.text.CountVectorizer(max_features = 10000, ngram_range=(1,2))
        self.vect_tfidf = feature_extraction.text.TfidfVectorizer(max_features = 10000, ngram_range=(1,2))
        self.stopwords_eng_ = nltk.corpus.stopwords.words('english')

    def preprocess(self, tweet):
        clean_tweet = self.clean_text(tweet, stemm= self.stemm_, lemm = self.lemm_ ,stop_words = self.stopwords_eng_)
        
        return clean_tweet

    def clean_text(self, observation, stemm, lemm, stop_words):
        #Clean tweets (removing punctuations and converting everything to lowercase)
        observation = re.sub(r'[^\$\w\s]', '', str(observation).lower().strip())
        observation = re.sub(r'^RT[\s]+', '', observation)
        observation = re.sub(r'https?:\/\/.*[\r\n]*', '', observation) # removing hyperlinks
        observation = re.sub(r'#', '', observation) #removing hash # sign

        #Tokenize (converting strings to lists)
        tokens_list = observation.split()
        tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
        tokens_list = tokenizer.tokenize(observation)
        #Remove the Stopwords
        if stop_words is not None:
            stop_text = []
            for word in tokens_list: 
                if (word not in stop_words and  # remove stopwords
        word not in punctuation ): # remove punctuation
                    word_ = ''
                    for char in word:
                        if (char not in punctuation and
                            char not in ['1','2','3','4','5','6','7','8','9','0','',' ']):
                            word_ = "".join((word_,char))
                    stop_text.append(word_)
                    tokens_list = stop_text.copy()
        if stemm == True:
            stem_text = []
            for word in tokens_list: # Go through every word in the tokens list
                # Init the Stemmer
                stem_word = self.stemmer_.stem(word)
                stem_text.append(stem_word)
            tokens_list = stem_text.copy()
        if lemm == True:
            lem_text = []
            for word in tokens_list: # Go through every word in the tokens list
                # Init the Wordnet Lemmatizer
                lem_word = self.lemmatizer_.lemmatize(word)
                lem_text.append(lem_word)
            tokens_list = lem_text.copy()
        
        tweets_clean = " ".join(tokens_list)
        return tweets_clean

@app.route('/predict',methods=['POST'])
def predict():
    #tweets_df= pd.read_csv("tweets_data.csv")
    #tweets_processed = TweetClassify(tweets_df, column_name='Text')
    #Preprocess the tweets 
    #X,y = tweets_processed.get_data()
    #vectorizer = vectorizer = feature_extraction.text.TfidfVectorizer(max_features = 10000, ngram_range=(1,2))
    #X = vectorizer.fit_transform(X)
    # Predictions
    #from sklearn.ensemble import RandomForestClassifier
    #rom sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #clf = RandomForestClassifier(random_state=42, n_jobs = -1)
    #clf.fit(X_train,y_train)
    #clf.score(X_test,y_test)
    if request.method == 'POST':
        message = request.form['message']
        clean_class = CleanTweet()
        clean_tweet = clean_class.preprocess(message)
        with open('classifier.sav','rb') as data:
            model = pickle.load(data)
        my_prediction = model.predict([clean_tweet])
    return render_template('result.html',prediction = my_prediction)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)


if __name__=='__main__':
    app.run(debug=True)

