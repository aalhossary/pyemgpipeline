# This example uses dataset1 to show the usage of class EMGMeasurement to
# process EMG data of one trial.

import os
import numpy as np
from matplotlib.figure import SubplotParams
import pyEMGproc as pep


# Setup example data
data_folder = r'..\..\data\dataset1'
data_filename = '3Asen.txt'
trial_name = 'Sit'
channel_names = ['rectus femoris', 'biceps femoris', 'vastus internus', 'semitendinosus']
frequency = 1000


# Load example data
with open(os.path.join(data_folder, data_filename)) as fp:
    collect_values = np.array([])
    lines = fp.readlines()
    for line in lines[7:]:  # first few lines are data description
        items = [float(e) for e in line.split('\t')[:4] if e != '']  # last column is not EMG data
        if len(items) != 4:  # last few rows might not have EMG data
            continue
        collect_values = np.concatenate((collect_values, np.array(items)))
    data = collect_values.reshape(-1, 4)
    print('data shape:', data.shape)


# Set EMG plot parameters
emg_plot_params = pep.EMGPlotParams(
    n_rows=4,
    fig_kwargs={
        'figsize': (5, 4),
        'subplotpars': SubplotParams(wspace=0, hspace=0.9),
    }
)


# Process EMG by using class EMGMeasurement
m = pep.EMGMeasurement(data, hz=frequency, channel_names=channel_names,
                       main_title=trial_name, emg_plot_params=emg_plot_params)
m.apply_dc_offset_remover()
m.apply_bandpass_filter()
m.apply_full_wave_rectifier()
m.apply_linear_envelope()
m.apply_end_frame_cutter()
max_amplitude = [0.043, 0.069, 0.364, 0.068]  # assume the MVC (maximum voluntary contraction) is known
m.apply_amplitude_normalizer(max_amplitude)
m.apply_segmenter(0, 999)  # can put tighter range, e.g., (9, 18.5), to extract the segment of interest


# plot the processed data
m.plot()
