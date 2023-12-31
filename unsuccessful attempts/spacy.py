# -*- coding: utf-8 -*-
"""SpaCy

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YG0aXO58iMQMX3vniifEgRV-6yekZMw0
"""

# Install spaCy and the French transformer-based model
!pip install -U spacy==3.7.0
# Import standard libraries
import pandas as pd
import numpy as np
import math
import bs4 as bs
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Import for text analytics
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim import corpora
import multiprocessing

# Import libraries for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Import libraries for hugginface
from transformers import pipeline
import gensim.downloader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn

from spacy.lang.fr.stop_words import STOP_WORDS
url = "https://raw.githubusercontent.com/Celso-Jorge-Sebastiao/UNIL_SBB/main/data/training_data.csv"
df_test = pd.read_csv("https://raw.githubusercontent.com/Celso-Jorge-Sebastiao/UNIL_SBB/main/data/unlabelled_test_data.csv")
df = pd.read_csv(url)

sentences = df["sentence"].tolist()
# Total number of words - over 600,000
words_number = df['sentence'].apply(lambda x: len(x.split(' '))).sum()
print(f'The sentences contain a total of {words_number} words.')
nlp = spacy.load("fr_core_news_sm")
print(df["sentence"].sample().values[0])
df.difficulty.value_counts()
#doc = nlp(sentences[0])
#print(doc.text)
#for token in doc:
 #   print(token.text, token.pos_, token.dep_)

base_rate = round(len(df[df.difficulty == "A1"]) / len (df), 4)
print(f'The base rate is: {base_rate*100:0.2f}%')
# Select features
X = df["sentence"] # Features we want to analyze
ylabels = df['difficulty']                # Labels we test against

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.1, random_state=1234)

spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
# Print total number of stopwords
print('Number of stopwords: %d' % len(spacy_stopwords))
# Print 20 stopwords
print('20 stopwords: %s' % list(spacy_stopwords)[:20])

# Define tokenizer function
def spacy_tokenizer(sentence):

    punctuations = string.punctuation
    stop_words = spacy.lang.fr.stop_words.STOP_WORDS

    # Create token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatize each token and convert each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Remove stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Remove anonymous dates and people
    mytokens = [ word.replace('xx/', '').replace('xxxx/', '').replace('xx', '') for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in ["xxxx", "xx", ""] ]

    # Return preprocessed list of tokens
    return mytokens

# Define vectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1,2), tokenizer = spacy_tokenizer )
# Define classifier
classifier = LogisticRegression(solver='lbfgs')

# Create pipeline
pipe = Pipeline([('vectorizer', tfidf),
                 ('classifier', classifier)])

# Fit model on training set

pipe.fit(X, ylabels)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate model

## Accuracy
accuracy_tfidf = round(accuracy_score(y_test, y_pred), 4)
print(f'The accuracy using TF-IDF is: {accuracy_tfidf*100:0.2f}%')

## Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

y_pred_test = pipe.predict(df_test["sentence"])
df_test["difficulty"] = y_pred_test
submission = df_test[["id","difficulty"]]
submission.to_csv('submission.csv', index=False)
from google.colab import files
# Assuming you have a file named 'predictions.csv'
files.download('submission.csv')