# splitData.py
# splits 'raw.mat' into training, test, and validation datasets

import scipy.io as spio
import numpy as np

# load raw data
mat = spio.loadmat('data/raw.mat', squeeze_me=True)
d = mat['d']
index = mat['Index']
classification = mat['Class']

# create 2D array for indices and classes, so
# each entry represents one spike, contianing
# its index and its class
spikes = np.array([index, classification])

# sort this array by the first row, so the spikes
# are in order of appearance
spikes = spikes[:, spikes[0].argsort()]

# define thresholds between datasets.
# Training goes from 0 to testStart, 
# test goes from testStart to validationStart,
# validation goes from validationStart to end.
testStart       = 864000
validationStart = 1152000

#--- split into datasets ---

# splice array up to testStart
trainingD       = d[:testStart]

# 'zip' to invert into rows such that the 0th column contains
# the index and the 1st column contains the classification,
# then loop over and output those whose index is less than
# testStart
trainingSpikes  = [s for s in zip(*spikes) if s[0] < testStart]

# same principle for test and validation
testD      = d[testStart:validationStart]
testSpikes = [s for s in zip(*spikes) if testStart <= s[0] < validationStart]

validationD = d[validationStart:]
validationSpikes = [s for s in zip(*spikes) if validationStart <= s[0]]

# save files
spio.savemat('data/training.mat', {
    'd': trainingD,
    'Index': [s[0] for s in trainingSpikes],
    'Class': [s[1] for s in trainingSpikes]
})
spio.savemat('data/test.mat', {
    'd': testD,
    'Index': [s[0] - testStart for s in testSpikes],
    'Class': [s[1] for s in testSpikes]
})
spio.savemat('data/validation.mat', {
    'd': validationD,
    'Index': [s[0] - validationStart for s in validationSpikes],
    'Class': [s[1] for s in validationSpikes]
})
