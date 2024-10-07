import numpy as np


class BufferToProcess:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.serveBuffer = np.empty((1, sampling_rate), dtype=np.complex64)
        self.buffer = np.empty((1, sampling_rate), dtype=np.complex64)
        self.availSamples = sampling_rate
        self.prevInd = 0
        self.isReady = False

    def add_to_buffer(self, rcv, num_rx_samples):
        endInd = self.prevInd + num_rx_samples
        rcvEnd = num_rx_samples

        if num_rx_samples > self.availSamples:
            endInd = self.prevInd + self.availSamples
            rcvEnd = self.availSamples
        if rcv.shape[1] == 0 or num_rx_samples == 0:
            return
        self.buffer[0, self.prevInd : endInd] = rcv[0, :rcvEnd]
        self.prevInd += rcvEnd
        self.availSamples -= rcvEnd

        if self.availSamples == 0:
            self.prevInd = 0
            self.serveBuffer = self.buffer
            self.isReady = True
            self.buffer[0, :num_rx_samples] = rcv[0, :num_rx_samples]
            self.availSamples = self.sampling_rate

    def get_buffer(self):
        self.isReady = False
        return self.serveBuffer

    def isReadyToRetr(self):
        return self.isReady
