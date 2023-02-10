import time
from brainflow.board_shim import BoardShim, LogLevels
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import brainflow_tools

'''
This script classifies mindfulness from EEG data using Brainflow.
'''

def main():
    # Connect to EEG
    board, board_descr, args  = brainflow_tools.connect_board()
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()
    board.start_stream(45000, args.streamer_params)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(5)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    # Collect data and extract feature
    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    feature_vector = bands[0]
    print(feature_vector)

    # Prepare Mindfulness Classifier from Brainflow (There is also a Focus Classifier look at Brainflow docs for more information)
    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()

    # Predict
    print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
    mindfulness.release()
    
if __name__ == "__main__":
    main()