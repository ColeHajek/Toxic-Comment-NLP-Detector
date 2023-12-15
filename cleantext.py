import re
from nltk.corpus import stopwords
from nltk import PorterStemmer, word_tokenize
import os
import pandas as pd
import numpy as np

# Uncomment if downloads are needed
#nltk.download('stopwords')
#nltk.download('punkt')

def cleanString(text):
    '''
    Removes stopwords, special characters, punctuation
    Turns all words into lowercase stemmed words
    '''
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    # Removing special characters and apostrophes
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)

    tokens = word_tokenize(text)
    stemmed_tokens = [porter.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(stemmed_tokens)


#used to make a csv file so that you dont have to clean the training data every time you run the program
def makeCleanCSVFiles(fn, fn_destination):
    '''
    Creates CSV file of cleaned data
    '''
    train_csv = os.path.join(os.getcwd(), fn_destination)
    train_data = pd.read_csv(fn)

    max_depth = int(0.5*len(train_data))

    # Process first half
    with open(train_csv, 'a', encoding='utf-8') as train_file:
        for index, row in train_data.iterrows():
            if index > max_depth:
                continue
            cleaned_text = cleanString(row['comment_text'])
            if len(cleaned_text) == 0:
                continue
            row['comment_text'] = cleaned_text
            
            # Convert row values to a comma-separated string
            row_str = ','.join(map(str, row.values.tolist())) + '\n'
            train_file.write(row_str)

    # Process second half
    with open(train_csv, 'a', encoding='utf-8') as train_file:
        for index, row in train_data.iterrows():
            if index <= max_depth:
                continue
            cleaned_text = cleanString(row['comment_text'])
            if len(cleaned_text) == 0:
                continue
            row['comment_text'] = cleaned_text
            
            
            # Convert row values to a comma-separated string
            row_str = ','.join(map(str, row.values.tolist())) + '\n'
            train_file.write(row_str)

input_fn = 'NAME OF .csv FILE TO CLEAN'
output_fn = 'NAME OF **EMPTY** OUTPUT FILE'

fileDirectory = os.path.join(os.getcwd(), input_fn)
fileDestination = os.path.join(os.getcwd(), output_fn)
makeCleanCSVFiles(fileDirectory,fileDestination)