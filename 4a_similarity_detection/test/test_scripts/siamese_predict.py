import keras.utils
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Lambda, Concatenate, Flatten, Dropout, Embedding, Bidirectional, Subtract, Multiply
import numpy as np
import GRADU_Siamese_Test_GEN as gen_test
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

# Test Data Generator
dataloader2 = gen_test.BinaryFileDocvec(root_dir=r'E:\Gradu\DataSet\Match\MatchedFiles_json_essential\Test') 
test_data_path = 'testdata.csv'
test_samples = dataloader2.load_samples(test_data_path)
num_test_samples = len(test_samples)
print ('number of armhf samples available to match: ', num_test_samples + 1)
batch_size2 = gen_test.Config.batch_size
test_datagen = dataloader2.data_generator(test_samples, batch_size=batch_size2, shuffle=False)

rnn = load_model('Siamese_RNN_15.h5')
rnn.summary()

scores = rnn.predict_generator(generator=test_datagen, steps=20, verbose=1, workers=1, max_q_size=1)
file1 = open("Predictions.txt","a")
for row in scores:
    np.savetxt(file1, row)




