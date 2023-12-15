import tools
import os
import sys

from keras.models import Sequential
from keras.layers import LSTM,Embedding,Dense,SpatialDropout1D
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.utils import set_random_seed

seed = 42
set_random_seed(seed)

# Load in training data
train_data_fn = 'NAME OF YOUR TRAINING DATA .csv FILE'
train_data_path = os.path.join(os.getcwd(), train_data_fn)
Xtr,ytr,Xte,yte = tools.preprocess_data(train_data_path)

# Path to save model weights and model output
checkpoint_fn = 'NAME OF YOUR CHECKPOINT .ckpt FILE'
results_fn = 'NAME OF YOUR RESULTS .txt FILE'
checkpoint_path = os.path.join(os.getcwd(), checkpoint_fn)
output_path = os.path.join(os.getcwd(), results_fn)

# True if loading model weights from file
# False if training model from scratch
load_model = True

# Set True to create file of model weights
save_model = False

# Model definition
num_words = 1000
embedding_dim = 100
drop_value = 0.2
max_len = 500

model = Sequential()
model.add(Embedding(num_words,embedding_dim,input_length=Xtr.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(10,dropout=drop_value,recurrent_dropout=drop_value))
model.add(Dense(6,activation='sigmoid'))

with open(output_path,'w') as f:
    sys.stdout = f

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=[BinaryAccuracy(),
                        Precision(),
                        Recall()])
    model.summary()

    if load_model:
        model.load_weights(checkpoint_path)
    else:
        # Training
        num_epochs = 1
        history = model.fit(Xtr,ytr,epochs=num_epochs,
                validation_data=(Xte,yte),verbose=2)

        if save_model ==True:
            model.save_weights(checkpoint_path)

    sys.stdout = sys.__stdout__

