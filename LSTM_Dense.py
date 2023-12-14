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
dataset_path = os.path.join(os.getcwd(), 'data\\50pruned_w_gen.csv')
Xtr,ytr,Xte,yte = tools.preprocess_data(dataset_path)

# Path to save model weights and model output
checkpoint_path = os.path.join(os.getcwd(), 'models\\LSTM_Dense_50_w_gen.ckpt')
output_path = os.path.join(os.getcwd(),'results\\50_w_gen_results.txt')


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

