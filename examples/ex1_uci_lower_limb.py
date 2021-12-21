# This example uses uci_lower_limb to show the usage of class EMGMeasurement to
# process EMG data of one trial.

import os
import pathlib
import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep

# Setup example data
repo_folder = pathlib.Path(__file__).parent.parent
data_folder = os.path.join(repo_folder, 'data', 'uci_lower_limb')
data_filename = '3Asen.txt'
trial_name = 'Sit'
channel_names = ['rectus femoris', 'biceps femoris', 'vastus internus', 'semitendinosus']
frequency = 1000


# Example data parsing function
def load_uci_lower_limb_txt(_filepath):
    with open(_filepath) as fp:
        collect_values = np.array([])
        lines = fp.readlines()
        for line in lines[7:]:  # first few lines are data description
            items = [float(e) for e in line.split('\t')[:4] if e != '']  # last column is not EMG data
            if len(items) != 4:  # last few rows might not have EMG data
                continue
            collect_values = np.concatenate((collect_values, np.array(items)))
    _data = collect_values.reshape(-1, 4)
    return _data


# Load example data
filepath = os.path.join(data_folder, data_filename)
data = load_uci_lower_limb_txt(filepath)
print('data shape:', data.shape)

# Set EMG plot parameters
emg_plot_params = pep.plots.EMGPlotParams(
    n_rows=4,
    fig_kwargs={
        'figsize': (5, 4),
        'subplotpars': SubplotParams(wspace=0, hspace=0.9),
    }
)


# Process EMG by using class EMGMeasurement
m = pep.wrappers.EMGMeasurement(data, hz=frequency, channel_names=channel_names,
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
