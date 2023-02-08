#import core pckgs
import xgboost as xgb
import streamlit as st
#import EDA pckgs
import pandas as pd
import numpy
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import email
import os
import re
import nltk

## Loadin the created model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

##Caching the model for faster loading
@st.cache

## data Cleaning............Creating Counters for striped email strings
class EmailToWordsCount(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, to_lowercase=True, remove_punc=True, do_stem=True):
        self.strip_headers = strip_headers
        self.to_lowercase = to_lowercase
        self.remove_punc = remove_punc
        self.do_stem = do_stem
        
        # To perform stemming
        self.stemmer = nltk.PorterStemmer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_word_counts = []
        for email in X:
            # text of the email
            plain = email_to_text(email)
            if plain is None:
                plain = "nothing"
            
            if self.to_lowercase:
                plain = plain.lower()
            
            if self.remove_punc:
                plain = plain.replace(".", "")
                plain = plain.replace(",", "")
                plain = plain.replace("!", "")
                plain = plain.replace("?", "")
                plain = plain.replace(";", "")
                
            word_counts = Counter(plain.split())
            if self.do_stem:
                # Stem the word, and add their counts
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    root_word = self.stemmer.stem(word)
                    stemmed_word_counts[root_word] += count
                word_counts = stemmed_word_counts
            
            X_word_counts.append(word_counts)
        return np.array(X_word_counts)

class WordCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    # train on list of word counts and build vocabulary
    def fit(self, X, y=None):
        total_word_counts = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_counts[word] += count
                
        # Build a vocabulary out of total most common
        self.most_common = total_word_counts.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: i for i, (word, count) in enumerate(self.most_common)}
    
        return self
    # Create the vector out of vocabulary
    def transform(self, X, y=None):
        X_new = np.zeros([X.shape[0], self.vocabulary_size + 1], dtype=int)
        
        # The vectors will contain additional column for counts of words
        # not captured in vocabulary
        for row, word_counts in enumerate(X):
            for word, count in word_counts.items():
                col = self.vocabulary_.get(word, self.vocabulary_size)
                X_new[row, col] += count
                
        return X_new
    
   #Creating a Pipeline for Data processing
email_to_countvector = Pipeline([
    ("emailToWords", EmailToWordsCount()), 
    ("wordCountVectorizer", WordCountVectorizer())
])

processed_input = email_to_countvector.fit_transform(input_email)

prediction = model.predict(processed_input)
return prediction

# Designing the app
def main():
    #Email classification
    st.title("Email Classifier Machine Learning App")
    st.subheader("With Streamlit")
    Activities = ["Classify Emails", "About"]
    choice = st.sidebar.selectbox("Select Activity", Activities)
    if choice == "Classify Emails":
        st.subheader("Classifying Emails with ML")
    elif choice == "About":
        st.subheader=("About App")
if__name __== '__main__':
    main()
