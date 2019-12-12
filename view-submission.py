import scipy.io as spio
from scipy import signal
from matplotlib import pyplot as plt

mat = spio.loadmat('data/training.mat', squeeze_me=True)
d = mat['d']

sampleFrequency = 25000

start = 10000
end = 15000

sos = signal.butter(2, [5, 5000], 'bp', fs=25000, output='sos')
dfilt = signal.sosfilt(sos, d)

plt.plot(d[start:end])
plt.plot(dfilt[start:end])
plt.legend(['d', 'dfilt'])
plt.show()
