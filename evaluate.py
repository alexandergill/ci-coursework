import tensorflow as tf
from tensorflow import keras
from preprocess import preprocessLabeled as ppl
import numpy as np

# load model from file
model = keras.models.load_model('model.h5')

# get list of detected spikes
spikes_test = ppl('data/test.mat')

# extract waveform data from each spike
# and save as N x 70 numpy array
data_test = np.array([s.waveform for s in spikes_test])

# save labels to N x 4 matrix where each row is zeros 
# apart from the correct label, which is 1
labels_test = np.array([
    # convert 'soft' category array (eg [0.01, 0.01, 0.99, 0.01])
    # into binary array (eg [0, 0, 1, 0])
    keras.utils.to_categorical(
        np.argmax(s.type),
        num_classes=4
    )
    for s in spikes_test # for each spike
])

# evaluate model
model.evaluate(data_test, labels_test, verbose=2)
