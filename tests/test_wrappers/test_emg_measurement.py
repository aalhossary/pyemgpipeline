from unittest import TestCase
import numpy as np
from pyemgpipeline.wrappers import EMGMeasurement


class TestEMGMeasurement(TestCase):
    def setUp(self):
        data = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                         [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                         [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                         [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        hz = 1000
        self.m = EMGMeasurement(data, hz)

    def tearDown(self):
        pass

    def test_apply_dc_offset_remover(self):
        self.m.apply_dc_offset_remover()
        actual = self.m.data
        desired = np.array([[8.405, -10.105], [29.105, -8.305], [42.005, -9.805], [51.405, -11.405], [27.605, -6.805],
                            [13.005, -4.005], [14.205, -1.305], [12.105, 7.895], [32.205, 2.995], [30.105, 7.595],
                            [25.505, 5.995], [12.705, 6.695], [-33.695, -0.105], [-68.195, 2.695], [-59.995, 4.195],
                            [-56.895, 8.195], [-40.995, 0.895], [-21.495, 5.695], [-6.595, 1.195], [-10.495, -2.205]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_bandpass_filter(self):
        bf_order = 4
        bf_cutoff_fq_lo = 10
        bf_cutoff_fq_hi = 450
        self.m.apply_bandpass_filter(bf_order=bf_order,
                                     bf_cutoff_fq_lo=bf_cutoff_fq_lo, bf_cutoff_fq_hi=bf_cutoff_fq_hi)
        actual = self.m.data
        desired = np.array([[-12.9626930, 3.0392905], [9.3485823, 4.1989569], [23.9736743, 2.5539775],
                            [35.1195602, -0.3272597], [12.6483173, 4.8784131], [0.4948079, 5.5820306],
                            [2.1619819, 9.8057245], [3.3989020, 16.0311044], [23.2768925, 13.4894575],
                            [24.9766021, 14.4783088], [19.9661507, 15.7359189], [11.0707771, 12.5597376],
                            [-35.7989373, 8.7509028], [-66.2563963, 7.7710270], [-58.5611268, 12.0242989],
                            [-51.5105100, 12.7371159], [-35.7001685, 7.5037866], [-12.8463616, 10.0108927],
                            [2.8389140, 6.3580431], [1.3684272, 2.1311555]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_full_wave_rectifier(self):
        self.m.apply_full_wave_rectifier()
        actual = self.m.data
        desired = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, 0.2], [39.5, 4.4],
                            [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                            [37.4, 17.2], [24.6, 17.9], [21.8, 11.1], [56.3, 13.9], [48.1, 15.4],
                            [45.0, 19.4], [29.1, 12.1], [9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_linear_envelope(self):
        le_order = 4
        le_cutoff_fq = 5
        self.m.apply_linear_envelope(le_order=le_order, le_cutoff_fq=le_cutoff_fq)
        actual = self.m.data
        desired = np.array([[3.2664113, -9.4276525], [3.3081268, -9.3487550], [3.3472161, -9.2732103],
                            [3.3835973, -9.2011398], [3.4172337, -9.1326544], [3.4481353, -9.0678542],
                            [3.4763497, -9.0068260], [3.5019513, -8.9496410], [3.5250378, -8.8963502],
                            [3.5457335, -8.8469814], [3.5641951, -8.8015377], [3.5806142, -8.7599969],
                            [3.5952102, -8.7223115], [3.6082107, -8.6884097], [3.6198237, -8.6581978],
                            [3.6302147, -8.6315597], [3.6395004, -8.6083555], [3.6477546, -8.5884216],
                            [3.6550211, -8.5715707], [3.6613283, -8.5575935]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_end_frame_cutter(self):
        n_end_frames = 3
        self.m.apply_end_frame_cutter(n_end_frames=n_end_frames)
        actual = self.m.data
        desired = np.array([[63.3, -0.2], [39.5, 4.4],
                            [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                            [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                            [-45.0, 19.4], [-29.1, 12.1]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_amplitude_normalizer(self):
        max_amplitude = [10, 20]
        self.m.apply_amplitude_normalizer(max_amplitude=max_amplitude)
        actual = self.m.data
        desired = np.array([[2.03, 0.055], [4.10, 0.145], [5.39, 0.07], [6.33, -0.01], [3.95, 0.22],
                            [2.49, 0.36], [2.61, 0.495], [2.40, 0.955], [4.41, 0.71], [4.20, 0.94],
                            [3.74, 0.86], [2.46, 0.895], [-2.18, 0.555], [-5.63, 0.695], [-4.81, 0.77],
                            [-4.50, 0.97], [-2.91, 0.605], [-0.96, 0.845], [0.53, 0.62], [0.14, 0.45]])
        np.testing.assert_allclose(actual, desired)

    def test_apply_segmenter(self):
        beg_ts = 0.002
        end_ts = 0.0085
        self.m.apply_segmenter(beg_ts=beg_ts, end_ts=end_ts)
        actual = self.m.data
        desired = np.array([[53.9, 1.4], [63.3, -0.2], [39.5, 4.4], [24.9, 7.2], [26.1, 9.9],
                            [24.0, 19.1], [44.1, 14.2]])
        np.testing.assert_allclose(actual, desired)
