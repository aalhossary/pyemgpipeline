from unittest import TestCase
import numpy as np
from pyemgpipeline import EMGMeasurementCollection


class TestEMGMeasurementCollection(TestCase):
    def setUp(self):
        hz = 1000

        # case: dict
        all_data_dict = {
            'T1': np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                            37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4]),
            'T2': np.array([1.1, 2.9, 1.4, -0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                            17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1, 16.9, 12.4, 9.0])
        }
        self.c_in_dict = EMGMeasurementCollection(all_data_dict, hz)

        # case: list
        all_data_list = [
            np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4]),
            np.array([1.1, 2.9, 1.4, -0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                      17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1, 16.9, 12.4, 9.0])
        ]
        self.c_in_list = EMGMeasurementCollection(all_data_list, hz)

    def tearDown(self):
        pass

    def test_apply_dc_offset_remover(self):
        # case: dict
        self.c_in_dict.apply_dc_offset_remover()
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([8.405, 29.105, 42.005, 51.405, 27.605,
                            13.005, 14.205, 12.105, 32.205, 30.105,
                            25.505, 12.705, -33.695, -68.195, -59.995,
                            -56.895, -40.995, -21.495, -6.595, -10.495]),
            'T2': np.array([-10.105, -8.305, -9.805, -11.405, -6.805,
                            -4.005, -1.305, 7.895, 2.995, 7.595,
                            5.995, 6.695, -0.105, 2.695, 4.195,
                            8.195, 0.895, 5.695, 1.195, -2.205])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_dc_offset_remover()
        actual = self.c_in_list.all_data
        desired = [
            np.array([8.405, 29.105, 42.005, 51.405, 27.605,
                      13.005, 14.205, 12.105, 32.205, 30.105,
                      25.505, 12.705, -33.695, -68.195, -59.995,
                      -56.895, -40.995, -21.495, -6.595, -10.495]),
            np.array([-10.105, -8.305, -9.805, -11.405, -6.805,
                      -4.005, -1.305, 7.895, 2.995, 7.595,
                      5.995, 6.695, -0.105, 2.695, 4.195,
                      8.195, 0.895, 5.695, 1.195, -2.205])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_apply_bandpass_filter(self):
        bf_order = 2
        bf_cutoff_fq_lo = 10
        bf_cutoff_fq_hi = 450

        # case: dict
        self.c_in_dict.apply_bandpass_filter(
            bf_order=bf_order, bf_cutoff_fq_lo=bf_cutoff_fq_lo, bf_cutoff_fq_hi=bf_cutoff_fq_hi)
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([-12.9626930, 9.3485823, 23.9736743, 35.1195602, 12.6483173,
                            0.4948079, 2.1619819, 3.3989020, 23.2768925, 24.9766021,
                            19.9661507, 11.0707771, -35.7989373, -66.2563963, -58.5611268,
                            -51.5105100, -35.7001685, -12.8463616, 2.8389140, 1.3684272]),
            'T2': np.array([3.0392905, 4.1989569, 2.5539775, -0.3272597, 4.8784131,
                            5.5820306, 9.8057245, 16.0311044, 13.4894575, 14.4783088,
                            15.7359189, 12.5597376, 8.7509028, 7.7710270, 12.0242989,
                            12.7371159, 7.5037866, 10.0108927, 6.3580431, 2.1311555])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_bandpass_filter(
            bf_order=bf_order, bf_cutoff_fq_lo=bf_cutoff_fq_lo, bf_cutoff_fq_hi=bf_cutoff_fq_hi)
        actual = self.c_in_list.all_data
        desired = [
            np.array([-12.9626930, 9.3485823, 23.9736743, 35.1195602, 12.6483173,
                      0.4948079, 2.1619819, 3.3989020, 23.2768925, 24.9766021,
                      19.9661507, 11.0707771, -35.7989373, -66.2563963, -58.5611268,
                      -51.5105100, -35.7001685, -12.8463616, 2.8389140, 1.3684272]),
            np.array([3.0392905, 4.1989569, 2.5539775, -0.3272597, 4.8784131,
                      5.5820306, 9.8057245, 16.0311044, 13.4894575, 14.4783088,
                      15.7359189, 12.5597376, 8.7509028, 7.7710270, 12.0242989,
                      12.7371159, 7.5037866, 10.0108927, 6.3580431, 2.1311555])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_apply_full_wave_rectifier(self):
        # case: dict
        self.c_in_dict.apply_full_wave_rectifier()
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                            37.4, 24.6, 21.8, 56.3, 48.1, 45.0, 29.1, 9.6, 5.3, 1.4]),
            'T2': np.array([1.1, 2.9, 1.4, 0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                            17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1, 16.9, 12.4, 9.0])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_full_wave_rectifier()
        actual = self.c_in_list.all_data
        desired = [
            np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, 21.8, 56.3, 48.1, 45.0, 29.1, 9.6, 5.3, 1.4]),
            np.array([1.1, 2.9, 1.4, 0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                      17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1, 16.9, 12.4, 9.0])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_apply_linear_envelope(self):
        le_order = 2
        le_cutoff_fq = 5

        # case: dict
        self.c_in_dict.apply_linear_envelope(le_order=le_order, le_cutoff_fq=le_cutoff_fq)
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([3.2664113, 3.3081268, 3.3472161, 3.3835973, 3.4172337,
                            3.4481353, 3.4763497, 3.5019513, 3.5250378, 3.5457335,
                            3.5641951, 3.5806142, 3.5952102, 3.6082107, 3.6198237,
                            3.6302147, 3.6395004, 3.6477546, 3.6550211, 3.6613283]),
            'T2': np.array([-9.4276525, -9.3487550, -9.2732103, -9.2011398, -9.1326544,
                            -9.0678542, -9.0068260, -8.9496410, -8.8963502, -8.8469814,
                            -8.8015377, -8.7599969, -8.7223115, -8.6884097, -8.6581978,
                            -8.6315597, -8.6083555, -8.5884216, -8.5715707, -8.5575935])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_linear_envelope(le_order=le_order, le_cutoff_fq=le_cutoff_fq)
        actual = self.c_in_list.all_data
        desired = [
            np.array([3.2664113, 3.3081268, 3.3472161, 3.3835973, 3.4172337,
                      3.4481353, 3.4763497, 3.5019513, 3.5250378, 3.5457335,
                      3.5641951, 3.5806142, 3.5952102, 3.6082107, 3.6198237,
                      3.6302147, 3.6395004, 3.6477546, 3.6550211, 3.6613283]),
            np.array([-9.4276525, -9.3487550, -9.2732103, -9.2011398, -9.1326544,
                      -9.0678542, -9.0068260, -8.9496410, -8.8963502, -8.8469814,
                      -8.8015377, -8.7599969, -8.7223115, -8.6884097, -8.6581978,
                      -8.6315597, -8.6083555, -8.5884216, -8.5715707, -8.5575935])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_apply_end_frame_cutter(self):
        n_end_frames = 3

        # case: dict
        self.c_in_dict.apply_end_frame_cutter(n_end_frames=n_end_frames)
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                            37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1]),
            'T2': np.array([-0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                            17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_end_frame_cutter(n_end_frames=n_end_frames)
        actual = self.c_in_list.all_data
        desired = [
            np.array([63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1]),
            np.array([-0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                      17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_find_max_amplitude_of_each_channel_across_trials(self):
        desired = np.array([63.3])

        # case: dict
        actual = self.c_in_dict.find_max_amplitude_of_each_channel_across_trials()
        np.testing.assert_allclose(actual, desired)

        # case: list
        actual = self.c_in_list.find_max_amplitude_of_each_channel_across_trials()
        np.testing.assert_allclose(actual, desired)

    def test_apply_amplitude_normalizer(self):
        max_amplitude = 10

        # case: dict
        self.c_in_dict.apply_amplitude_normalizer(max_amplitude=max_amplitude)
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([2.03, 4.10, 5.39, 6.33, 3.95, 2.49, 2.61, 2.40, 4.41, 4.20,
                            3.74, 2.46, -2.18, -5.63, -4.81, -4.50, -2.91, -0.96, 0.53, 0.14]),
            'T2': np.array([0.11, 0.29, 0.14, -0.02, 0.44, 0.72, 0.99, 1.91, 1.42, 1.88,
                            1.72, 1.79, 1.11, 1.39, 1.54, 1.94, 1.21, 1.69, 1.24, 0.90])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        self.c_in_list.apply_amplitude_normalizer(max_amplitude=max_amplitude)
        actual = self.c_in_list.all_data
        desired = [
            np.array([2.03, 4.10, 5.39, 6.33, 3.95, 2.49, 2.61, 2.40, 4.41, 4.20,
                      3.74, 2.46, -2.18, -5.63, -4.81, -4.50, -2.91, -0.96, 0.53, 0.14]),
            np.array([0.11, 0.29, 0.14, -0.02, 0.44, 0.72, 0.99, 1.91, 1.42, 1.88,
                      1.72, 1.79, 1.11, 1.39, 1.54, 1.94, 1.21, 1.69, 1.24, 0.90])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])

    def test_apply_segmenter(self):
        # case: dict
        all_beg_ts = {'T1': 0.002, 'T2': 0.0015}
        all_end_ts = {'T1': 0.008, 'T2': 0.0055}
        self.c_in_dict.apply_segmenter(all_beg_ts=all_beg_ts, all_end_ts=all_end_ts)
        actual = self.c_in_dict.all_data
        desired = {
            'T1': np.array([53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1]),
            'T2': np.array([1.4, -0.2, 4.4, 7.2])
        }
        self.assertEqual(actual.keys(), desired.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], desired[k])

        # case: list
        all_beg_ts = [0.002, 0.0015]
        all_end_ts = [0.008, 0.0055]
        self.c_in_list.apply_segmenter(all_beg_ts=all_beg_ts, all_end_ts=all_end_ts)
        actual = self.c_in_list.all_data
        desired = [
            np.array([53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1]),
            np.array([1.4, -0.2, 4.4, 7.2])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])
