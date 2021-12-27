import numpy as np
import pyemgpipeline as pep

# prepare data  # 1 trial  # 1 channel in each trial, shape = (40,)
data = np.array([14.191, 15.106, 7.172, 4.425, 8.698, 12.36, 9.308, 11.749, 9.308, 13.58,
                 9.613, 7.172, 3.51, 8.698, 5.646, 9.003, 6.866, 5.341, 10.223, 10.834,
                 4.12, 5.035, 8.087, 8.087, 6.866, 4.425, -1.068, 3.51, 18.463, 8.392,
                 6.561, 12.97, 12.054, 9.003, 5.035, 6.561, 3.815, 3.204, 5.341, -0.458])
hz = 1000

# initialize EMGMeasurement
m = pep.wrappers.EMGMeasurement(data, hz=hz)

# apply seven processing steps
m.apply_dc_offset_remover()
m.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=5, bf_cutoff_fq_hi=420)
m.apply_full_wave_rectifier()
m.apply_linear_envelope(le_order=4, le_cutoff_fq=8)
m.apply_end_frame_cutter(n_end_frames=2)
m.apply_amplitude_normalizer(max_amplitude=1.5)
m.apply_segmenter(beg_ts=0.01, end_ts=0.04)

# plot final result
m.plot()
