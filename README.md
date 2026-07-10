# Email/SMS Spam Classifier

A simple machine learning web app that detects whether a message is spam or not. Built using Python, Scikit-learn, and Streamlit.
    ## Live Demo
      Try it here: https://email-sms-spam-classifier-ze5tnmyqpevjvchvrtz8lc.streamlit.app/


      
## What it does

You type or paste any email or SMS message into the app, and it predicts whether the message is spam or a normal (ham) message. It uses natural language processing to clean the text and a trained ML model to make the prediction.

## How it works

1. The input text is cleaned and preprocessed (lowercasing, removing stopwords and punctuation, and stemming words).
2. The cleaned text is converted into numerical features using a TF-IDF vectorizer.
3. A trained classification model predicts whether the message is spam or not.

## Tech used

- Python
- Streamlit (for the web interface)
- Scikit-learn (for the ML model)
- NLTK (for text preprocessing)

## Project files

- `app.py` - main Streamlit app
- `model.pkl` - trained classification model
- `vectorizer.pkl` - saved TF-IDF vectorizer
- `requirements.txt` - list of required packages

## Note

This project was built as a hands-on way to learn text classification and understand how NLP preprocessing works together with a trained ML model in a real, usable app.
## Running it locally

Clone this repository, then install the required packages:
