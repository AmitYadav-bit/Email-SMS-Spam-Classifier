from unittest import result
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

import pickle 

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
import nltk.stem.porter

# Data reprocessing

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
       y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

ps = nltk.stem.porter.PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_text = st.text_area("Enter the email/SMS content:")

if st.button("Predict"):

    # preprocess

    transform_sms = transform_text(input_text)

    # vectorize
    vector_input = tfidf.transform([transform_sms]) 

    # predict

    result = model.predict(vector_input)[0]

    # Display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")