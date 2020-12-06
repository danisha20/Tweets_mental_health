# Classifying Tweets for Mental Health 

Detecting Depression From Tweets

# Data Collection: 
The Model was trainned with 1,600,000 tweets obtained from https://www.kaggle.com/kazanova/sentiment140. 

# Labelling: 
The Tweets were flagged as having a relation with depression if they contained the word “depression”, its derivatives or  similar as per Ferran et al. [1]. The list includes the words: overwhelmed, exhausted, distressed, anxiety, anxious, tired, low, sad, cry, nervous, empty, worried, insomnia, demotivated, lonely, desperate.

# Preprocessing
Natural language preprocessing techniques such as noise reduction (removing special characters, URLs, extra spaces, etc...), normalization ( lower-case transformation, punctuation removal, contractions expansion), stop words elimination, lemmatization, and tokenization were performed to clean the tweets. 

# Model Selection
Several supervised machine learning algorithms were tested to determine which one was the better at discerning messages with a link to depression than those that had no link. For this,  three algorithms were evaluated: Naive Bayes, Random Forest, and KNN and tested them with two different vectorization techniques: Bag of Words(BoW) and TF-IDF and with two n-grams strategies: bi-grams and tri-grams.

# Results

The best combination obtained from this experimentation was: **BoW + Tri-grams + Random Forest** obtaining an accuracy of 97% and Area Under the Curve of 98%. 

# Deployment 
TogetherAlways App 


[1] Leis, A., Ronzano, F., Mayer, M. A., Furlong, L. I., & Sanz, F. (2019). Detecting signs of depression in tweets in Spanish: behavioral and linguistic analysis. Journal of medical Internet research, 21(6), e14199.

