from ctypes import sizeof
import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import TclError

# use this backend to display in separate Tk window
def pcm24to32(data, channels=1):
    if len(data) % 3 != 0:
        raise ValueError('Size of data must be a multiple of 3 bytes')

    out = np.zeros(len(data) // 3, dtype='<i4')
    out.shape = -1, channels
    temp = out.view('uint8').reshape(-1, 4)
    columns = slice(None, -1)
    temp[:, columns] = np.frombuffer(data, dtype='uint8').reshape(-1, 3)
    return out


# constants
CHUNK = 4800             # samples per frame
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
x = np.arange(0, CHUNK, 1)

# create a line object with random data
line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)

# basic formatting for the axes
ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_ylim(-33000, 33000)
ax.set_xlim(0, CHUNK)
plt.setp(ax, xticks=[0, CHUNK/2, CHUNK], yticks=[-33000, 0, 33000])

# show the plot
plt.show(block=False)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    
    # binary data
    data = stream.read(CHUNK)  
    
    data32 = pcm24to32(data)
    data32 = data32.reshape(1,-1)
    data_np = np.array(data32)
    #print(len(data_np))
    data_np = data_np * np.hanning(len(data_np))
    #data_np = np.array(struct.unpack(str(CHUNK) + 'h', data))
    # convert data to integers, make np array, then offset it by 127
    #data_int = struct.unpack(str(4 * CHUNK) + 'B', data)
    #print(len(data_int))
    # create np array and offset by 128
    
    
    line.set_ydata(data_np)
    
    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
        
    except TclError:
        
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

