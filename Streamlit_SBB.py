#pip install streamlit PyPDF2 spacy scikit-learn gensim transformers torch
#transformers : C:\Users\celso\pythonProject3\venv\Scripts\python.exe -m pip install --upgrade pip
import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import math
#import bs4 as bs
import urllib.request

#import matplotlib.pyplot as plt

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
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


data=pd.read_csv("https://raw.githubusercontent.com/Celso-Jorge-Sebastiao/UNIL_SBB/main/training_data.csv")
from transformers import AutoTokenizer, CamembertModel
import torch

tokenizer = AutoTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained("camembert-base")
def calculate_embedding(tokens):
    if tokens:
        # Tokenisation avec le tokenizer BERT
        encoded_input = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)

        # Passage des tokens à travers le modèle BERT pour obtenir les embeddings
        with torch.no_grad():
            output = model(**encoded_input)

        # Récupérer les embeddings de la couche d'attention (ou d'une autre couche selon vos besoins)
        last_hidden_states = output.last_hidden_state

        # Calculer l'embedding moyen de la phrase
        sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
        return sentence_embedding.tolist()
    else:
        return None
# Appliquer la fonction à la colonne 'tokenized_sentence' de votre DataFrame
data['embedding'] = data['sentence'].apply(calculate_embedding)
classifier_LR = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
classifier_kernel = SVC(kernel='linear', C=1.1, random_state=42, probability=True)

# Fit classifiers on the training data
classifier_LR.fit(data['embedding'].tolist(), data['difficulty'])
classifier_kernel.fit(data['embedding'].tolist(), data['difficulty'])

# Create a soft voting classifier
classifier_voting = VotingClassifier(estimators=[
    ('LR', classifier_LR),
    ('kernel', classifier_kernel),
], voting='soft')
classifier_voting.fit(data['embedding'].tolist(), data['difficulty'])
classifier_stacking = StackingClassifier(
    estimators=[
        ('LR', classifier_LR),
        ('kernel', classifier_kernel),
        ('voting',classifier_voting)
    ],
    final_estimator=classifier_LR
)


classifier_stacking.fit(data['embedding'].tolist(), data['difficulty'])
try:
    import os
except:
    raise Exception("Please note that some cells will not work, you will need to create a .py file manually")


selected_option = st.selectbox('Select an option:', ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
# Display the selected option
st.write(f'You selected: {selected_option}')



# Download the Punkt tokenizer for sentence splitting (run only once)
nltk.download('punkt')

# Get the PDF file path from user input
pdf_path = st.text_input("Enter the path to the PDF file")

# Streamlit app title
st.title("Book level prediction")

# Check if the user provided a path and the file exists
if pdf_path and st.button("Read PDF"):
    pdf_path = pdf_path.strip('"')  # Remove any surrounding quotes from the path
    try:
        # Open the PDF file and read its content
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            num_pages = len(pdf_reader.pages)

            # Extract sentences from all pages using nltk's sent_tokenize
            sentences = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                sentences.extend(sent_tokenize(page_text))

            # Create a DataFrame with the sentences
            df = pd.DataFrame({'sentence': sentences})
            df['sentence'] = df['sentence'].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True).replace('–', '', regex=True)
            # Supprimer la première et dernière ligne du DataFrame
            df = df.drop(df.index[-1])
            df = df.drop(df.index[0])
            # Supprimer les lignes contenant ")" ou "(" dans la colonne "Phrases"
            df = df[-df['sentence'].str.contains('[\(\)]')]
            # Supprimer les lignes ne contenant qu'un seul mot dans la colonne "Phrases"
            df = df[df['sentence'].apply(lambda x: len(x.split()) > 2)]
            # Réinitialiser l'index du DataFrame
            df = df.reset_index(drop=True)
            print(df.head(),df.shape)
            df['embedding'] = df['sentence'].apply(calculate_embedding)
            df['difficulty'] = classifier_stacking.predict(df['embedding'].tolist())

            output = df['difficulty'].value_counts().to_dict()
            print(output)
            total_count = sum(output.values())
            output_percentages = {key: (value / total_count) * 100 for key, value in output.items()}
            somme = 0
            liste = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            level = 'A1'
            for item in liste:
                somme += output_percentages[item]
                print(item, somme)
                if somme > 80:
                    break
                else:
                    level = item

            st.write(f"The level of the text is : {level}")

    except FileNotFoundError:
        st.error(f"The specified file does not exist: {pdf_path}")