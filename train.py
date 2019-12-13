# train.py
# train a model on the data contained in 'data/training.mat'

# import everything we need
import tensorflow as tf
from tensorflow import keras
from preprocess import preprocessLabeled as ppl
import numpy as np

# extract and process spikes from training data
spikes_train = ppl('data/training.mat')

# save waveforms to N x 70 matrix, such that each row
# is one waveform
data_train = np.array([s.waveform for s in spikes_train])

# save labels to N x 4 matrix where each row is zeros 
# apart from the correct label, which is 1
labels_train = np.array([
    # convert 'soft' category array (eg [0.01, 0.01, 0.99, 0.01])
    # into binary array (eg [0, 0, 1, 0])
    keras.utils.to_categorical(
        np.argmax(s.type), num_classes=4
    )
    for s in spikes_train # for each spike
])

# create neural network object
model = keras.models.Sequential([
    # input layer
    tf.keras.layers.Dense(70, activation='relu'),
    # hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    # create drouput to avoid over-fitting
    tf.keras.layers.Dropout(0.2),
    # output layer
    tf.keras.layers.Dense(4, activation='softmax')
])
# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model on training data
model.fit(data_train, labels_train, epochs=50)

# save trained model to file
model.save('model.h5')
