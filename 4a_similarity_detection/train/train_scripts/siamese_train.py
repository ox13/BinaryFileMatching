import keras.utils
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Lambda, Concatenate, Flatten, Dropout, Embedding, Bidirectional, Subtract, Multiply
import numpy as np
import GRADU_Siamese_Train_GEN as gen
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# Train and Validate Data Generator
dataloader = gen.BinaryFileDocvec(root_dir=r'TRAIN_FILES_PATH') 
train_data_path = 'traindata.csv'
samples = dataloader.load_samples(train_data_path)
# num_train_samples = len(samples)
# print ('number of armhf samples available to match: ', num_train_samples + 1)
train_samples, validation_samples = train_test_split(samples, test_size=0.20)

batch_size = gen.Config.batch_size
# train_datagen = dataloader.data_generator(train_samples, batch_size=batch_size)
validation_datagen = dataloader.data_generator(validation_samples, batch_size=batch_size)

datagen = dataloader.data_generator(train_samples, batch_size=batch_size)



def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


# Model: fileLength = size of the input, docvecLength = size of desired output
def buildModel(sentLength, docvecLength):

   
    armhf_base = Input(shape=(None, sentLength),name="armhf_base")
    rasp_match = Input(shape=(None, sentLength),name="rasp_match")
    
    
    armhf_Layer_1 = Bidirectional(LSTM(64, return_sequences=False, return_state=False, dropout=0.2))(armhf_base)
    arm_vector = armhf_Layer_1
        
    rasp_Layer_1 = Bidirectional(LSTM(64, return_sequences=False, return_state=False, dropout=0.2))(rasp_match)
    rasp_vector = rasp_Layer_1
    

    # measure the cosin distance between the two output vectors
    distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([arm_vector, rasp_vector])

    diff = Subtract()([arm_vector, rasp_vector])
    diff_sq = Multiply()([diff, diff])
   
    arm_vec_sq = Multiply()([arm_vector, arm_vector])
    rasp_vec_sq = Multiply()([rasp_vector, rasp_vector])
    sq_diff = Subtract()([arm_vec_sq, rasp_vec_sq])

    conc = Concatenate(axis=-1)([distance,sq_diff,diff_sq])

    in_dense = Dense(100, activation='relu')(conc)
    drop_match = Dropout(0.2)(in_dense)
    match = Dense(1, activation='sigmoid')(drop_match)

    model = Model(inputs=[armhf_base,rasp_match], outputs=[match])
    myAdam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=myAdam, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

rnn = buildModel(768, 192)
rnn.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
rnn.fit_generator(generator=datagen, epochs=100, steps_per_epoch=80, validation_data=validation_datagen, validation_steps=20, callbacks=[monitor])

rnn.save('Siamese_RNN_01.h5')


