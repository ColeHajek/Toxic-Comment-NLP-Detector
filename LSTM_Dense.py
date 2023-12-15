import tools
import os
import sys

from keras.models import Sequential
from keras.layers import LSTM,Embedding,Dense,SpatialDropout1D
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.utils import set_random_seed

seed = 42
set_random_seed(seed)


def run_model(Xtr,ytr,Xte,yte,load_model=False):
    # Model definition
    num_words = 1000
    embedding_dim = 100
    drop_value = 0.2

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
                            Precision(),Recall()])
        model.summary()

        if load_model:
            model.load_weights(checkpoint_path)
        else:
            # Training
            num_epochs = 1
            history = model.fit(Xtr,ytr,epochs=num_epochs,
                    validation_data=(Xte,yte),verbose=2)

            # Save weights
            model.save_weights(checkpoint_path)

        sys.stdout = sys.__stdout__

    return history

train_data_fns = ['data/50pruned_w_gen.csv','data/60pruned_w_gen.csv',
                     'data/70pruned_w_gen.csv','data/80pruned_w_gen.csv',
                     'data/90pruned_w_gen.csv']

test_data_fn = 'data/testclean.csv'

checkpoint_fns = ['model/50pruned_w_gen.ckpt','model/60pruned_w_gen.ckpt',
                     'model/70pruned_w_gen.ckpt','model/80pruned_w_gen.ckpt',
                     'model/90pruned_w_gen.ckpt']

results_fns = ['results/50pruned_w_gen.txt','results/60pruned_w_gen.txt',
               'results/70pruned_w_gen.txt','results/80pruned_w_gen.txt',
               'results/90pruned_w_gen.txt']

for i in range(3,5):
    # Load in training data
    train_data_path = os.path.join(os.getcwd(), train_data_fns[i])
    Xtr,ytr = tools.preprocess_data(train_data_path)

    # Load in test data
    test_data_path = os.path.join(os.getcwd(), test_data_fn)
    Xte,yte = tools.preprocess_data(test_data_path)

    # Path to save/load model weights and model output
    checkpoint_path = os.path.join(os.getcwd(), checkpoint_fns[i])
    output_path = os.path.join(os.getcwd(), results_fns[i])

    # Run model
    run_model(Xtr,ytr,Xte,yte,False)



