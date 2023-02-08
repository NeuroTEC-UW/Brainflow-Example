import argparse
import time
import numpy as np
# from beepy import beep
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pickle import dump, load

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
    print(board_descr)
    return board, board_descr, args
 
def main():
    training_length = 30 #seconds
    '''
    Connect to EEG
    '''
    board, board_descr, args  = connect_eeg()

    board.prepare_session()
    ch_indexes = board_descr['eeg_channels']
    sfreq = int(board_descr['sampling_rate'])
    ch_names = board.get_eeg_names(board.board_id)
    '''
    Collect EEG Data
    '''
    board.start_stream()
    print("Calibrating ...")
    time.sleep(5)
    board.get_board_data()

    # Collect eeg data for fist hand
    print("Hold a fist with your hand ...")
    time.sleep(training_length)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data_0 = board.get_board_data()  # get all data and remove it from internal buffer

    #Collect eeg data for open hand
    # beep(sound="ping")
    print("Finished. Now prepare for next collection period")
    time.sleep(5)
    print("Hold a open hand ...")

    board.get_board_data()
    time.sleep(training_length)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data_1 = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()
    # beep(sound="ping")

    dump(data_0, open("./data/" + args.name +"data_0.pkl", 'wb'))
    dump(data_1, open("./data/" + args.name +"data_1.pkl", 'wb'))

    # # Apply filters
    # data_0 = eeg_tools.apply_noise_filter(data_0, ch_indexes, sfreq)
    # data_1 = eeg_tools.apply_noise_filter(data_1, ch_indexes, sfreq)
    # data_0 = data_0[ch_indexes[0]:ch_indexes[-1] + 1, 2*sfreq:]
    # data_1 = data_1[ch_indexes[0]:ch_indexes[-1] + 1, 2*sfreq:]

    # # Plot egg channels before
    # plt.figure(figsize =[10,15])
    # eeg_tools.plot_eeg(data_0, len(ch_indexes))
    # eeg_tools.plot_eeg(data_1, len(ch_indexes))
    # plt.show()

if __name__ == "__main__":
    main()