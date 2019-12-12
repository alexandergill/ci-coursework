import tensorflow as tf
from tensorflow import keras
from preprocess import preprocessLabeled as pp
import numpy as np

model = keras.models.load_model('model.h5')

# get test data the same way as above
spikes_test = pp('data/test.mat')
data_test = np.array([s.waveform for s in spikes_test])
labels_test = np.array([keras.utils.to_categorical(np.argmax(s.type), num_classes=4) for s in spikes_test])

# evaluate model
model.evaluate(data_test, labels_test, verbose=2)
