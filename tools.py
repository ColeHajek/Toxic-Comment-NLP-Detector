import pandas as pd
import numpy as np
#import torch

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


from sklearn.model_selection import train_test_split
#import nltk
import re
#nltk.download('stopwords')

def preprocess_data(file_path):
    #get data from csv files
    data = pd.read_csv(file_path)
    # Extract features and labels
    tokenizer = Tokenizer(num_words=500,lower=True)
    tokenizer.fit_on_texts(data['comment_text'].values)
    X = tokenizer.texts_to_sequences(data['comment_text'].values)
    X = pad_sequences(X,maxlen=250)


    Y = pd.get_dummies(data[['toxic', 'severe_toxic',
                                 'obscene', 'threat', 'insult',
                                 'identity_hate']]).values


    Xtr,Xte,ytr,yte = train_test_split(X,Y, test_size=0.1)

   
    return Xtr,ytr,Xte,yte

def tokenize(x,num_words=100):
    tokenizer =Tokenizer(num_words=num_words,lower=False)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    return pad_sequences(x, padding='post')


def clean_text(text):
    '''
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
    '''

def one_hot_encode(labels, num_classes=6):
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

