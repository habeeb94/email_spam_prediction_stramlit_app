import streamlit as st
import streamlit_authenticator as stauth
#import core pckgs
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog

#import EDA pckgs
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import email
import os
import re
import nltk
import yaml
import matplotlib



# Set up tkinter
root = tk.Tk()
root.withdraw()
export DISPLAY =:0.0

with open('Yaml.yaml') as file: #file = open('Yaml.yaml')
    config = yaml.load(file, Loader=yaml.SafeLoader)

#authenticator objects
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

## Loadin the created model
model = xgb.XGBRegressor()
model.load_model('xgb2_model.json')

##Caching the model for faster loading
@st.cache


#Converting html to text
def html_to_text(email) -> str:
    try:
        soup = BeautifulSoup(email.get_payload(), "html.parser")
        plain = soup.text.replace("=\n", "")
        plain = re.sub(r"\s+", " ", plain)
        return plain.strip()
    except:
        return "nothing"

def get_key(val, my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key.any()
email_labels = {"ham":0, "spam": 1}

#htlml to word converter
def email_to_text(email):
    text_content = ""
    for part in email.walk():
        part_content_type = part.get_content_type()
        if part_content_type not in ['text/plain', 'text/html']:
            continue
        if part_content_type == 'text/plain':
            text_content += part.get_payload()
        else:
            text_content += html_to_text(part)
    return text_content


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


#App loging page
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

if authentication_status == True:
    # Designing the app for Classification
    def main():
        #Email classification
        st.title("Email Classifier ML App")
        st.subheader("With Streamlit")
        Activities = ["Classify Emails", "About"]
        choice = st.sidebar.selectbox("Select Activity", Activities)
        if choice == "Classify Emails":
            st.subheader("Classifying Emails with ML")
            # Make folder picker dialog appear on top of other windows
            root.wm_attributes('-topmost', 1)
            # Folder picker button
            st.title('Folder Picker')
            st.write('Please select a folder:')
            clicked = st.button('Folder Picker')
            if clicked:
                filename = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
            #filename = st.text_input('Enter a file path:')
            #st.text_input('Enter a folder path that contain email files:')
            try:
                filename_data = [name for name in sorted(os.listdir(filename))]

                def parse_email(fname):
                    directory = filename
                    with open(os.path.join(directory,fname), "rb") as fp:
                        return email.parser.BytesParser().parse(fp)
                new_data = [parse_email(name) for name in filename_data]
                #try:
                    #with open(filename, "rb") as input:
                        #st.text(input.read()) 
                        #return email.parser.BytesParser().parse(input)
                #except FileNotFoundError:
                    #st.error('File not found.')

                #mail_text = st.text_area("Paste email here")
                if st.button("Classify"):
                    data = email_to_countvector.fit_transform(new_data)

                    prediction = model.predict(data).round()
                    #final_result = get_key(prediction, email_labels)
                    st.write(prediction)
            except:
                st.error("Please enter the correct file path")        
        elif choice == "About":
            st.subheader=("About App")
    if __name__== '__main__':
        main()

