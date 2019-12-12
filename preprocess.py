import scipy.io as spio
from scipy import signal
import numpy as np
from tqdm import tqdm

# class to store each spike
class Spike:

    # how many samples to each side of the peak
    # should be stored in the waveform
    border = 35

    def __init__(self):
        self.waveform = None
        self.type     = None
        self.position = None
        self.centre   = None

# function to bandpass filter the waveform
def bandpass(d):
    # create filter with cutoff frequencies 5 & 5 000
    sos = signal.butter(2, [5, 5000], 'bp', fs=25000, output='sos')
    
    # apply filter and return
    return signal.sosfilt(sos, d)

# preprocess labeled data for training
def preprocessLabeled(filepath):
    # empty list to store spikes in
    spikes = []

    # load data from 'filepath'
    mat = spio.loadmat(filepath, squeeze_me=True)

    # extract the data from the file
    d = mat['d']
    indices = mat['Index']
    classification = mat['Class']

    waveform = d

    # detect peaks in the waveform
    peaks, _ = signal.find_peaks(
        waveform,
        height=2,
        distance=5,prominence=1
    )

    # loop over peaks
    for peak in tqdm(
        peaks,
        desc='creating preprocessed dataset from %s' % filepath
    ):

        # get the location in the array 'indices' of the largest
        # index below the peak
        indexLoc = np.argmax([i for i in indices if i <= peak])

        # create new Spike object
        spike = Spike()
        # save the window either side of the spike
        spike.waveform = np.interp( # np.interp normalises the data
            waveform[peak - spike.border:peak + spike.border],
            (-1.5, 12.5), # normalise from these values 
            (0, 1)        # to these values
        )
        # save the ground-truth position of the spike
        spike.position = indices[indexLoc]  - peak + spike.border
        # create label array (all 0.01 except ground truth,
        # which is 0.99)
        spike.type = np.zeros(4) + 0.01
        spike.type[classification[indexLoc] - 1] = 0.99

        # check that the waveform is the right length
        # it can be the wrong length if the spike is too close
        # to either end of the waveform sample
        if len(spike.waveform) == 70:
            # append this spike to the list
            spikes.append(spike)
        # ignore if the spike is malformed

    return spikes

# preprocess unlabaled data
def preprocessUnlabeled(filepath):
    # empty list to store spikes in
    spikes = []

    # load data from 'filepath'
    mat = spio.loadmat(filepath, squeeze_me=True)

    # extract the data from the file
    d = mat['d']

    # apply filter to waveform
    waveform = bandpass(d)

    # detect peaks in the waveform
    peaks, _ = signal.find_peaks(
        waveform,
        height=2,
        distance=5,prominence=1
    )

    # loop over peaks
    for peak in tqdm(
        peaks,
        desc='creating preprocessed dataset from %s' % filepath
    ):

        # create new Spike object
        spike = Spike()
        # save the window either side of the spike
        spike.waveform = np.interp( # np.interp normalises the data
            waveform[peak - spike.border:peak + spike.border],
            (-1.5, 12.5), # normalise from these values 
            (0, 1)        # to these values
        )

        # check that the waveform is the right length
        # it can be the wrong length if the spike is too close
        # to either end of the waveform sample
        if len(spike.waveform) == 70:
            # append this spike to the list
            spikes.append(spike)
        # ignore if the spike is malformed

    return spikes
