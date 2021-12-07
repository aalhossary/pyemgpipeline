from unittest import TestCase
import numpy as np
from pyemgpipeline.processors import BandpassFilter


class TestBandpassFilter(TestCase):
    def setUp(self):
        hz = 1000
        bf_order = 2
        bf_cutoff_fq_lo = 10
        bf_cutoff_fq_hi = 450
        self.bandpass_filter = BandpassFilter(
            hz=hz, bf_order=bf_order, bf_cutoff_fq_lo=bf_cutoff_fq_lo, bf_cutoff_fq_hi=bf_cutoff_fq_hi)

    def tearDown(self):
        pass

    def test_apply__case_x_1dim(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])
        actual = self.bandpass_filter.apply(x)
        desired = np.array([-12.9626930, 9.3485823, 23.9736743, 35.1195602, 12.6483173,
                            0.4948079, 2.1619819, 3.3989020, 23.2768925, 24.9766021,
                            19.9661507, 11.0707771, -35.7989373, -66.2563963, -58.5611268,
                            -51.5105100, -35.7001685, -12.8463616, 2.8389140, 1.3684272])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_1channel(self):
        x = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                      [37.4], [24.6], [-21.8], [-56.3], [-48.1], [-45.0], [-29.1], [-9.6], [5.3], [1.4]])
        actual = self.bandpass_filter.apply(x)
        desired = np.array([[-12.9626930], [9.3485823], [23.9736743], [35.1195602], [12.6483173],
                            [0.4948079], [2.1619819], [3.3989020], [23.2768925], [24.9766021],
                            [19.9661507], [11.0707771], [-35.7989373], [-66.2563963], [-58.5611268],
                            [-51.5105100], [-35.7001685], [-12.8463616], [2.8389140], [1.3684272]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_2channel(self):
        x = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                      [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                      [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                      [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        actual = self.bandpass_filter.apply(x)
        desired = np.array([[-12.9626930, 3.0392905], [9.3485823, 4.1989569], [23.9736743, 2.5539775],
                            [35.1195602, -0.3272597], [12.6483173, 4.8784131], [0.4948079, 5.5820306],
                            [2.1619819, 9.8057245], [3.3989020, 16.0311044], [23.2768925, 13.4894575],
                            [24.9766021, 14.4783088], [19.9661507, 15.7359189], [11.0707771, 12.5597376],
                            [-35.7989373, 8.7509028], [-66.2563963, 7.7710270], [-58.5611268, 12.0242989],
                            [-51.5105100, 12.7371159], [-35.7001685, 7.5037866], [-12.8463616, 10.0108927],
                            [2.8389140, 6.3580431], [1.3684272, 2.1311555]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__assertion_raise(self):
        # require: n_samples > (2 * bf_order + 1) * 3
        with self.assertRaises(AssertionError):
            x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                          37.4, 24.6, -21.8, -56.3, -48.1])  # n_samples = 15
            self.bandpass_filter.apply(x)  # (2 * bf_order + 1) * 3 = 15

    def test_get_parameter_str(self):
        actual = self.bandpass_filter.get_param_values_in_str()
        desired = 'hz = 1000, bf_order = 2, bf_cutoff_fq_lo = 10, bf_cutoff_fq_hi = 450'
        self.assertEqual(actual, desired)
