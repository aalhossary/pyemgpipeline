import numpy as np
from matplotlib.figure import SubplotParams
import pyemgpipeline as pep

# prepare data  # 2 trials  # 2 channel in each trial, shape = (30, 2)
all_data = [
    np.array([[8.698, -9.613], [7.172, -2.594], [3.51, -5.951], [8.087, -2.899], [5.035, -2.289],
              [10.529, -0.763], [-2.289, 4.73], [4.12, 1.068], [0.153, -4.12], [12.665, -9.918],
              [9.613, -7.782], [15.106, -9.918], [9.003, -12.665], [7.477, -22.43], [2.899, -10.223],
              [5.646, -11.749], [-5.646, -8.698], [1.373, -9.918], [3.204, -5.951], [6.866, -6.866],
              [3.51, -5.951], [10.529, -7.172], [13.275, -8.087], [11.444, -13.275], [12.054, -13.275],
              [11.139, -6.561], [9.308, -7.782], [8.087, -7.477], [18.463, -1.068], [1.984, -4.12]]),
    np.array([[-14.801, -12.97], [-21.21, -10.834], [-27.313, 7.477], [-42.878, -5.646], [-42.573, -5.646],
              [-35.859, -9.613], [-42.268, -10.834], [-23.041, -10.529], [-16.022, -9.613], [-12.97, -11.749],
              [-9.613, -12.665], [-22.125, -16.937], [-23.346, -11.749], [-42.268, -8.087], [-53.864, -9.613],
              [-41.047, -7.477], [-27.924, -16.632], [-28.839, -8.698], [-19.684, -8.392], [-17.242, -4.73],
              [6.866, -8.698], [37.995, 3.204], [78.279, -3.815], [102.694, -3.204], [132.296, -9.003],
              [141.756, -9.308], [113.375, -4.12], [102.388, -12.97], [86.823, -6.866], [78.584, -15.717]])]
hz = 1000

# initialize DataProcessingManager
mgr = pep.wrappers.DataProcessingManager()

# set data, parameters, and processors
mgr.set_data_and_params(all_data, hz=hz, trial_names=['Trial 1', 'Trial 2'], channel_names=['ch1', 'ch2'],
                        emg_plot_params=pep.plots.EMGPlotParams(
                            n_cols=2,
                            fig_kwargs={'figsize': (10, 2), 'subplotpars': SubplotParams(top=0.8)}))
mgr.set_end_frame_cutter(pep.processors.EndFrameCutter(n_end_frames=3))
mgr.set_amplitude_normalizer(pep.processors.AmplitudeNormalizer())
mgr.set_segmenter(pep.processors.Segmenter(),
                  all_beg_ts=[0,0],
                  all_end_ts=[0.018, 999])
mgr.show_current_processes_and_related_params()

# apply processing steps
c = mgr.process_all(is_plot_processing_chain=True)
