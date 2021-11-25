# This example uses dataset2 to show the usage of class DataProcessingManager
# to process EMG data of multiple trials under a fixed sequence of standard
# steps. Data of multiple trials are stored in a list for demonstration purpose.

import os
import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep


# Setup example data
data_folder = r'..\..\data\dataset2'
data_filenames = ['1_raw_data_11-08_21.03.16.txt', '2_raw_data_11-10_21.03.16.txt']
all_trial_names = ['trial 1', 'trial 2']
channel_names = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']
frequency = 1000


# Load example data
all_timestamp = []
all_data = []
for fn in data_filenames:
    data = np.genfromtxt(os.path.join(data_folder, fn), delimiter='\t', skip_header=1)
    print('raw data shape (including time, EMG, and class):', data.shape)
    all_timestamp.append(data[:, 0] / frequency)
    all_data.append(data[:, 1:9])
print(f'Load {len(all_data)} data files')


# Set EMG plot parameters
emg_plot_params = pep.EMGPlotParams(
    n_rows=4,
    n_cols=2,
    fig_kwargs={
        'figsize': (7, 5),
        'subplotpars': SubplotParams(wspace=0.2, hspace=0.8),
    }
)


# Process EMG by using class DataProcessingManager
mgr = pep.DataProcessingManager()
mgr.set_data_and_params(all_data, hz=frequency,
                        all_timestamp=all_timestamp, channel_names=channel_names,
                        all_main_titles=all_trial_names, emg_plot_params=emg_plot_params)
mgr.set_bandpass_filter(pep.BandpassFilter(hz=frequency, bf_cutoff_fq_hi=495))  # can change processor's parameter
mgr.set_amplitude_normalizer(pep.AmplitudeNormalizer())  # add non-default processor
mgr.show_current_processes_and_related_params()  # display current setting
c = mgr.process_all(is_plot_processing_chain=True, k_for_plot=0)  # execute processing steps in the standard order,
                                                    # plot the processing chain of trial k=0,
                                                    # and return an instance of the EMGMeasurementCollection class


# # save processed data as csv files
# dest_dir = r'C:/Users/user1/Downloads'
# all_csv_path = [os.path.join(dest_dir, e + '.csv') for e in all_trial_names]
# c.export_csv_all(all_csv_path)