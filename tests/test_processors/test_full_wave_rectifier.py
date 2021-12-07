from unittest import TestCase
import numpy as np
from pyemgpipeline.processors import FullWaveRectifier


class TestFullWaveRectifier(TestCase):
    def setUp(self):
        self.full_wave_rectifier = FullWaveRectifier()

    def tearDown(self):
        pass

    def test_apply__case_x_1dim(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])
        actual = self.full_wave_rectifier.apply(x)
        desired = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                            37.4, 24.6, 21.8, 56.3, 48.1, 45.0, 29.1, 9.6, 5.3, 1.4])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_1channel(self):
        x = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                      [37.4], [24.6], [-21.8], [-56.3], [-48.1], [-45.0], [-29.1], [-9.6], [5.3], [1.4]])
        actual = self.full_wave_rectifier.apply(x)
        desired = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                            [37.4], [24.6], [21.8], [56.3], [48.1], [45.0], [29.1], [9.6], [5.3], [1.4]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_2channel(self):
        x = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                      [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                      [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                      [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        actual = self.full_wave_rectifier.apply(x)
        desired = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, 0.2], [39.5, 4.4],
                            [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                            [37.4, 17.2], [24.6, 17.9], [21.8, 11.1], [56.3, 13.9], [48.1, 15.4],
                            [45.0, 19.4], [29.1, 12.1], [9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        np.testing.assert_allclose(actual, desired)

    def test_get_parameter_str(self):
        actual = self.full_wave_rectifier.get_param_values_in_str()
        desired = 'No parameters'
        self.assertEqual(actual, desired)