import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


# plt.style.use('ggplot')

class Graph:
    def __init__(self, queue):
        self.update_speed_ms = 50
        self.window_size = 4
        self.n_of_plots = 1
        self.curves = list()
    
        self.queue = queue

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='Channel Sounder',size=(800, 600))
        # self.win.setBackground('b')
        
        self.initPowerPlot() 
        

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()


    def initPowerPlot(self):
        p = self.win.addPlot(row=0,col=0)
        p.setLabel("left", "Power dB")
        p.setLabel("bottom", "Samples")
        p.setXRange(0, 50, padding=0)
        p.setYRange(-50, 0, padding=0)
        p.setMenuEnabled('left', False)
        p.setMenuEnabled('bottom', False)
        pen = pg.mkPen(color=(255, 0, 0))

        p.setTitle('Power Plot')
        curve = p.plot(pen=pen)

        self.curves.append(curve)

    def update(self):
        for ind in range(self.n_of_plots):
            self.curves[ind].setData(self.queue.get())

        self.app.processEvents()

