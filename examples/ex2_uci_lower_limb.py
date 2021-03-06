# This example uses uci_lower_limb to show the usage of class EMGMeasurementCollection
# to process EMG data of multiple trials. Data of multiple trials are stored in a list.

import os
import pathlib
import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep

# Setup example data
repo_folder = pathlib.Path(__file__).parent.parent
data_folder = os.path.join(repo_folder, 'data', 'uci_lower_limb')
data_filenames = ['5Nsen.txt', '5Npie.txt', '5Nmar.txt']
trial_names = ['Sit', 'Stand', 'Gait']
channel_names = ['rectus femoris', 'biceps femoris', 'vastus internus', 'semitendinosus']
sample_rate = 1000


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
all_data = []
for i in range(len(data_filenames)):
    filepath = os.path.join(data_folder, data_filenames[i])
    data = load_uci_lower_limb_txt(filepath)
    print('data shape:', data.shape)
    all_data.append(data)
print(f'Load {len(all_data)} data files')


# Set EMG plot parameters
emg_plot_params = pep.plots.EMGPlotParams(
    n_rows=4,
    fig_kwargs={
        'figsize': (8, 5),
        'subplotpars': SubplotParams(wspace=0, hspace=0.9),
    },
    line2d_kwargs={
        'color': 'green'
    }
)


# Process EMG by using class EMGMeasurementCollection
c = pep.wrappers.EMGMeasurementCollection(all_data, hz=sample_rate, trial_names=trial_names,
                                          channel_names=channel_names, emg_plot_params=emg_plot_params)
c.apply_dc_offset_remover()
c.apply_bandpass_filter()
c.apply_full_wave_rectifier()
c.apply_linear_envelope()
c.apply_end_frame_cutter()
max_amplitude = c.find_max_amplitude_of_each_channel_across_trials()
print('max_amplitude:', max_amplitude)
c.apply_amplitude_normalizer(max_amplitude)
all_beg_ts = []
all_end_ts = []
for k in range(len(all_data)):
    all_beg_ts.append(0)
    all_end_ts.append(999)
c.apply_segmenter(all_beg_ts, all_end_ts)  # can put tighter ranges to extract the segments of interest


# plot the processed data
c[1].plot()  # plot the trial 'Stand'
c.plot(is_overlapping_trials=True,
       main_title='All trials', cycled_colors=['r','b','k'],
       legend_kwargs={'loc':'right', 'bbox_to_anchor':(1.2, 0.5)},
       axes_pos_adjust=(0, 0, 0.8, 1))  # plot all trials
