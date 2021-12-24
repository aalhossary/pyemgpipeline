import numpy as np
import pyemgpipeline as pep

# prepare data  # 2 trials  # 1 channel in each trial, shape = (40,)
all_data = [np.array([-5.646, 12.665, 16.937, 1.678, -27.008, -15.411, -3.51, 11.139,
                      10.529, 1.068, -2.289, 1.984, -26.703, -54.475, -61.799, -37.385,
                      -17.242, -0.763, 19.073, 49.897, 62.104, 69.734, 60.884, 67.597,
                      35.554, 7.477, 8.087, 2.594, -12.054, 16.327, 55.696, 85.298,
                      84.992, 28.534, 23.041, 45.625, 49.897, 6.256, -57.221, -62.409]),
            np.array([-3.815, -1.068, -3.51, 0.153, 7.477, 5.341, 19.989, 12.665,
                      5.341, 0.153, 3.204, 12.97, 4.73, 12.665, 8.087, 14.191,
                      12.36, 14.496, 8.087, 14.191, 3.815, 4.12, -0.763, 2.289,
                      -4.12, -7.477, -10.529, -5.035, 6.866, 9.308, 2.594, 11.749,
                      1.984, -6.256, -5.951, -7.477, -8.392, -7.172, -5.646, -5.646])]
hz = 1000

# initialize EMGMeasurementCollection
c = pep.wrappers.EMGMeasurementCollection(all_data, hz=hz, all_main_titles=['Trial 1', 'Trial 2'])

# apply seven processing steps
c.apply_dc_offset_remover()
c.apply_bandpass_filter(bf_order=2, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=450)
c.apply_full_wave_rectifier()
c.apply_linear_envelope(le_order=2, le_cutoff_fq=6)
c.apply_end_frame_cutter(n_end_frames=2)
c.apply_amplitude_normalizer(max_amplitude=38.3)
c.apply_segmenter(all_beg_ts=[0, 0], all_end_ts=[999, 999])

# plot final result
c.plot()
