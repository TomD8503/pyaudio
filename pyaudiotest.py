from ctypes import sizeof
import pyaudio
import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import time
from tkinter import TclError

matplotlib.use('TkAgg')


def wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(nchannels, -1)
    return result

def GenerateFrequencyBands(NumberOfOctaves, BandsPerOctave, StartFreq):
    multiplier = np.emath.power(2, 1/BandsPerOctave)
    FreqArray = np.array([StartFreq])
    for i in range(NumberOfOctaves*BandsPerOctave):
        FreqArray = np.append(FreqArray, (FreqArray[i]*multiplier))
    return FreqArray



# constants
CHUNK = 4096             # samples per frame
FORMAT = pyaudio.paInt24     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 48000                 # samples per second

# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# variable for plotting

frequencies = int((CHUNK/2)+1)
x = np.arange(0, frequencies, 1)

# create a line object with random data
line, = ax.plot(x, np.random.rand(frequencies), '-', lw=2)

# basic formatting for the axes
ax.set_title('frequency amplitude')
ax.set_xlabel('frequency')
ax.set_ylabel('amplitude')
ax.set_ylim(0, 2**33)
ax.set_xlim(0, frequencies)
plt.setp(ax, xticks=[0, 400, frequencies], yticks=[0, 4000, 33000])

#test freq banding
NumberOfOctaves = 10
BandsPerOctave = 6
StartFreq = 20
FreqArray = GenerateFrequencyBands(NumberOfOctaves, BandsPerOctave, StartFreq)
print(FreqArray)


# show the plot
plt.show(block=False)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    
    # binary data
    data = stream.read(CHUNK)  
    data = wav2array(CHANNELS,3,data)
    data = data[:,0]
    size = len(data)
    window = np.hanning(size)
    data_hann = data * window
    #print("window size:{}".format(window.shape))
    #print("data size:{}".format(data.shape))
    #print("data_hann size:{}".format(data_hann.shape))
    dft = np.abs(scipy.fft.rfft(data_hann))
    

    
    
    line.set_ydata(dft)
    
    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1

        
        
    except (TclError):
        
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

