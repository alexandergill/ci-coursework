import tensorflow as tf
from tensorflow import keras
from preprocess import preprocessLabeled as pp
import numpy as np

# extract and process spikes from training data
spikes_train = pp('data/training.mat')

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

model = keras.models.Sequential([
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data_train, labels_train, epochs=50)

# get test data the same way as above
spikes_test = pp('data/test.mat')
data_test = np.array([s.waveform for s in spikes_test])
labels_test = np.array([keras.utils.to_categorical(np.argmax(s.type), num_classes=4) for s in spikes_test])

# evaluate model
model.evaluate(data_test, labels_test, verbose=2)

model.save('model.h5')
