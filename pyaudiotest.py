
import pyaudio
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import scipy


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # constants
        self.CHUNK = 4096*2             # samples per frame
        FORMAT = pyaudio.paInt24     # audio format (bytes per sample?)
        self.CHANNELS = 1                 # single channel for microphone
        RATE = 48000   
        # pyaudio class instance
        p = pyaudio.PyAudio()
        # stream object to get data from microphone
        self.stream = p.open(
        format=FORMAT,
        channels=self.CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=self.CHUNK)

        #frequencies = int((CHUNK/2)+1)
        #print(frequencies)
        #FrequencyBands = np.arange(0, frequencies, 1)
        self.frequency = scipy.fft.rfftfreq(self.CHUNK,1/RATE)
        magnitude = np.random.rand(self.CHUNK)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setLogMode(x=True, y=True)
        self.pen = pg.mkPen(color=(255, 0, 0), width=2, style=QtCore.Qt.SolidLine)
        self.graphWidget.setYRange(0, 10)
        self.graphWidget.showGrid(x=True, y=True)
        #self.graphWidget.plot(frequency, magnitude)
    
        
        data = self.stream.read(self.CHUNK)  
        data = wav2array(self.CHANNELS,3,data)
        data = data[:,0]
        size = len(data)
        #print(size)
        window = np.hanning(size)
        data_hann = data * window
        magnitude = np.abs(scipy.fft.rfft(data_hann))
        self.graphWidget.clear()
        self.MyPlot = self.graphWidget.plot(self.frequency, magnitude, pen=self.pen)

         # ... init continued ...
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def update_plot_data(self):
        data = self.stream.read(self.CHUNK)  
        data = wav2array(self.CHANNELS,3,data)
        data = data[:,0]
        size = len(data)
        #print(size)
        window = np.hanning(size)
        data_hann = data * window
        magnitude = np.abs(scipy.fft.rfft(data_hann))
        #self.graphWidget.clear()
        #self.graphWidget.plot(self.frequency, magnitude, pen=self.pen)
        self.MyPlot.setData(self.frequency, magnitude)  # Update the data.
            


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


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


if __name__ == '__main__':
    main()

#test freq banding
NumberOfOctaves = 10
BandsPerOctave = 6
StartFreq = 20
FreqArray = GenerateFrequencyBands(NumberOfOctaves, BandsPerOctave, StartFreq)
print(FreqArray)
