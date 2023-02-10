import argparse
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pickle import dump, load
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import logging

import pyqtgraph as pg

def connect_eeg():
    '''Connect to eeg
    '''
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--model', type=str, help='which classifier to run', required=False,
                        default="")
    parser.add_argument('--name', type=str, help='which classifier to run', required=False,
                        default="")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    board = BoardShim(args.board_id, params)
    board_descr = board.get_board_descr(args.board_id)
    return board, board_descr, args

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='BrainFlow Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()
 
def main():
    '''
    Connect to EEG
    '''
    board, board_descr, args  = connect_eeg()

    try:
        print(board_descr)
        board.prepare_session()
        board.start_stream(450000, args.streamer_params)
        Graph(board)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board.is_prepared():
            logging.info('Releasing session')
            board.release_session()

    # board.prepare_session()
    # ch_indexes = board_descr['eeg_channels']
    # sfreq = int(board_descr['sampling_rate'])
    # ch_names = board.get_eeg_names(board.board_id)
    # '''
    # Collect EEG Data
    # '''
    # board.start_stream()
    # print("Calibrating ...")
    # time.sleep(5)
    # board.get_board_data()

    # # Collect eeg data for fist hand
    # print("Hold a fist with your hand ...")
    # time.sleep(training_length)
    # # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    # data_0 = board.get_board_data()  # get all data and remove it from internal buffer

    # #Collect eeg data for open hand
    # # beep(sound="ping")
    # print("Finished. Now prepare for next collection period")
    # time.sleep(5)
    # print("Hold a open hand ...")

    # board.get_board_data()
    # time.sleep(training_length)
    # # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    # data_1 = board.get_board_data()  # get all data and remove it from internal buffer
    # board.stop_stream()
    # board.release_session()
    # # beep(sound="ping")

    # dump(data_0, open("./data/" + args.name +"data_0.pkl", 'wb'))
    # dump(data_1, open("./data/" + args.name +"data_1.pkl", 'wb'))

    # # # Apply filters
    # # data_0 = eeg_tools.apply_noise_filter(data_0, ch_indexes, sfreq)
    # # data_1 = eeg_tools.apply_noise_filter(data_1, ch_indexes, sfreq)
    # # data_0 = data_0[ch_indexes[0]:ch_indexes[-1] + 1, 2*sfreq:]
    # # data_1 = data_1[ch_indexes[0]:ch_indexes[-1] + 1, 2*sfreq:]

    # # # Plot egg channels before
    # # plt.figure(figsize =[10,15])
    # # eeg_tools.plot_eeg(data_0, len(ch_indexes))
    # # eeg_tools.plot_eeg(data_1, len(ch_indexes))
    # # plt.show()

if __name__ == "__main__":
    main()