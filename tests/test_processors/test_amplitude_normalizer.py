from unittest import TestCase
import numpy as np
from pyemgpipeline.processors import AmplitudeNormalizer


class TestAmplitudeNormalizer(TestCase):
    def setUp(self):
        self.amplitude_normalizer = AmplitudeNormalizer()

    def tearDown(self):
        pass

    def test_apply__case_x_1dim(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])
        desired = np.array([2.03, 4.10, 5.39, 6.33, 3.95, 2.49, 2.61, 2.40, 4.41, 4.20,
                            3.74, 2.46, -2.18, -5.63, -4.81, -4.50, -2.91, -0.96, 0.53, 0.14])

        divisor1 = 10
        actual1 = self.amplitude_normalizer.apply(x, divisor=divisor1)
        np.testing.assert_allclose(actual1, desired)

        divisor2 = [10]
        actual2 = self.amplitude_normalizer.apply(x, divisor=divisor2)
        np.testing.assert_allclose(actual2, desired)

        divisor3 = np.array([10])
        actual3 = self.amplitude_normalizer.apply(x, divisor=divisor3)
        np.testing.assert_allclose(actual3, desired)

    def test_apply__case_x_2dim_1channel(self):
        x = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                      [37.4], [24.6], [-21.8], [-56.3], [-48.1], [-45.0], [-29.1], [-9.6], [5.3], [1.4]])
        desired = np.array([[2.03], [4.10], [5.39], [6.33], [3.95], [2.49], [2.61], [2.40], [4.41], [4.20],
                            [3.74], [2.46], [-2.18], [-5.63], [-4.81], [-4.50], [-2.91], [-0.96], [0.53], [0.14]])

        divisor1 = 10
        actual1 = self.amplitude_normalizer.apply(x, divisor=divisor1)
        np.testing.assert_allclose(actual1, desired)

        divisor2 = [10]
        actual2 = self.amplitude_normalizer.apply(x, divisor=divisor2)
        np.testing.assert_allclose(actual2, desired)

        divisor3 = np.array([10])
        actual3 = self.amplitude_normalizer.apply(x, divisor=divisor3)
        np.testing.assert_allclose(actual3, desired)

    def test_apply__case_x_2dim_2channel(self):
        x = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                      [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                      [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                      [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        desired = np.array([[2.03, 0.055], [4.10, 0.145], [5.39, 0.07], [6.33, -0.01], [3.95, 0.22],
                            [2.49, 0.36], [2.61, 0.495], [2.40, 0.955], [4.41, 0.71], [4.20, 0.94],
                            [3.74, 0.86], [2.46, 0.895], [-2.18, 0.555], [-5.63, 0.695], [-4.81, 0.77],
                            [-4.50, 0.97], [-2.91, 0.605], [-0.96, 0.845], [0.53, 0.62], [0.14, 0.45]])

        divisor1 = [10, 20]
        actual1 = self.amplitude_normalizer.apply(x, divisor=divisor1)
        np.testing.assert_allclose(actual1, desired)

        divisor2 = np.array([10, 20])
        actual2 = self.amplitude_normalizer.apply(x, divisor=divisor2)
        np.testing.assert_allclose(actual2, desired)

    def test_apply__assertion_raise(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])

        # require: divisor > 0
        with self.assertRaises(AssertionError):
            divisor = 0
            self.amplitude_normalizer.apply(x, divisor=divisor)

    def test_get_parameter_str(self):
        actual = self.amplitude_normalizer.get_parameter_str()
        desired = 'No parameters'
        self.assertEqual(actual, desired)
