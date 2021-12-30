#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:11:36 2021

@author: ralph ZOGO
"""
import pickle
import re 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import spacy
import warnings
warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_lg")
#nltk.download('stopwords')
stop_words = stopwords.words('english')
#nltk.download()
from bs4 import BeautifulSoup

def post_preprocessing(text):
    ''' Fonction pemettant le prétraitement du post afin qu'il soit utilisable par les algorithmes de
        machine learning, 
       en entrée on a la question(text) pour obtenir
       en sortie un ensemble de mots de clés'''
    
    
    # Extraction du texte contenu dans le "Body" des balises 
    b_soup = BeautifulSoup(text, 'lxml')
    text = b_soup.get_text() 
    
    
    # Suppression des caractères jugés inutiles
    text = re.sub('\S*@\S*\s?', '', text) 
    text = re.sub('\s+', ' ', text)
    text = re.sub("\'", "", text)
    
    # Tokenization avec suppression de ponctuation s'il y en a
    tokenizer = RegexpTokenizer(r'\w+')
    tok = tokenizer.tokenize(text)
    
    # Suppression des stopwords
    filtered_words = [word for word in tok if word not in stopwords.words('english')]
    
    # Lemmatization
    lemma= WordNetLemmatizer()
    lemmatized_word=[]
    for word in filtered_words :
        lemmatized_word.append(lemma.lemmatize(word))
        
    # Transformation de la liste de mots en chaîne de mots pour une exploitation ultérieure sur le modèle de machine learning
    processed_post = " ".join(lemmatized_word)
    
    return processed_post

# Etape 2: Phase de prédiction des tags via l'algorithme de machine learning approprié

def predicted_tags(txt):
    ''' Fonction qui permet la prédiction de tags par le biais de l'algorithme
        du OneVsRestClassifier(LogisticRegression(C= 10), en entrée on a un ensemble de mots clés
        et en sortie une prédiction des mots clés adéquats pour tagguer une question sur Stack Overflow'''
        
    a = txt.split(" ")
    file1 = open("vecto.pkl", "rb")
    tfid = pickle.load(file1)
    file1.close()
        
    b = tfid.transform(a)
        
    file2 = open("model.pkl", "rb")
    model = pickle.load(file2)
    file2.close()
        
    matrix = model.predict(b)
    preds = tfid.inverse_transform(matrix)
    tags = preds[0]  
    return tags
