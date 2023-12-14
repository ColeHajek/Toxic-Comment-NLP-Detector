#import torch
#from torch import nn
#from torch.utils.data import DataLoader
import tools
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM,Embedding,Dropout,Flatten,Dense,GlobalAveragePooling1D,SpatialDropout1D
from keras.metrics import BinaryAccuracy, Precision, Recall
import numpy as np
import os
import pandas as pd

dataset_path = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 178\\Final Project\\data\\train.csv"
Xtr,ytr,Xte,yte = tools.preprocess_data(dataset_path)

# Define model

num_words = 1000
embedding_dim = 100
drop_value = 0.2
max_len = 500

model = Sequential()
model.add(Embedding(num_words,embedding_dim,input_length=Xtr.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(10,dropout=drop_value,recurrent_dropout=drop_value))
model.add(Dense(6,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[BinaryAccuracy(),
                     Precision(),
                     Recall()])
model.summary()

checkpoint_path = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 178\\Final Project\\models\\LSTM_Dense_training_cleaned.ckpt"

# True if loading model weights from file
# False if trainign model from scratch
load_model = False

# Set True to create file of model weights
save_model = True

if load_model==False:
    # Training
    num_epochs = 1
    history = model.fit(Xtr,ytr,epochs=num_epochs,
            validation_data=(Xte,yte),verbose=2)

    if save_model ==True:
        model.save_weights(checkpoint_path)

if load_model == True:
    model.load_weights(checkpoint_path)

train_results = model.evaluate(Xtr,ytr)
test_results = model.evaluate(Xte, yte)
print(f'Train accuracy: {train_results[1]:10.2f}')
print(f'Test accuracy: {test_results[1]:10.2f}')
