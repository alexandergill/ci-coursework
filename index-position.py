from preprocess import preprocessLabeled as ppl
from matplotlib import pyplot as plt
import numpy as np

# get list of detected spikes
spikes = ppl('data/training.mat')

# array of possible offsets
offsets = range(0, 50)
tally   = np.zeros(len(offsets))

# loop over spikes
for spike in spikes:
    # get offset
    offset = spike.border - spike.position

    # add to tally
    tally[offset] += 1

# plot histogram
plt.plot(offsets, tally)
plt.xlabel('offset')
plt.ylabel('frequency')
plt.show()
