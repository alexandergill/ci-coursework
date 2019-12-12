from tensorflow import keras
from preprocess import preprocessUnlabeled as ppu
from scipy import io
import numpy as np

# most common offset from actual index to spike
MODE_OFFSET = 8

# get list of detected spikes
spikes = ppu('data/submission.mat')

# extract waveform data from each spike
# and save as N x 70 numpy array
data = np.array([s.waveform for s in spikes])

# load model from file
model = keras.models.load_model('model.h5')

# use model to predict classes from each spike
result = model.predict(data)

# empty arrays for indices and classes
indices = np.zeros(len(result))
classes = np.zeros(len(result))

# loop over results
for i in range(len(result)):
    # save the spike's predicted class
    classes[i] = np.argmax(result[i] + 1)
    # save the index of the spike
    indices[i] = spikes[i].centre - MODE_OFFSET

# get my candidate number
CANDNUM = int(input('enter candidate number: '))

# save data as a .mat file
io.savemat('%d.mat' % CANDNUM, {
    'Index': indices,
    'Class': classes
})

# print some info
print('number of peaks detected: %d' % len(spikes))
