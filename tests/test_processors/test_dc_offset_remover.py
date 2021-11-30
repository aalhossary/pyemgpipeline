from unittest import TestCase
import numpy as np
from pyemgpipeline import DCOffsetRemover


class TestDCOffsetRemover(TestCase):
    def setUp(self):
        self.dc_offset_remover = DCOffsetRemover()

    def tearDown(self):
        pass

    def test_apply__case_x_1dim(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])
        actual = self.dc_offset_remover.apply(x)
        desired = np.array([8.405, 29.105, 42.005, 51.405, 27.605,
                            13.005, 14.205, 12.105, 32.205, 30.105,
                            25.505, 12.705, -33.695, -68.195, -59.995,
                            -56.895, -40.995, -21.495, -6.595, -10.495])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_1channel(self):
        x = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                      [37.4], [24.6], [-21.8], [-56.3], [-48.1], [-45.0], [-29.1], [-9.6], [5.3], [1.4]])
        actual = self.dc_offset_remover.apply(x)
        desired = np.array([[8.405], [29.105], [42.005], [51.405], [27.605],
                            [13.005], [14.205], [12.105], [32.205], [30.105],
                            [25.505], [12.705], [-33.695], [-68.195], [-59.995],
                            [-56.895], [-40.995], [-21.495], [-6.595], [-10.495]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_2channel(self):
        x = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                      [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                      [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                      [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        actual = self.dc_offset_remover.apply(x)
        desired = np.array([[8.405, -10.105], [29.105, -8.305], [42.005, -9.805], [51.405, -11.405], [27.605, -6.805],
                            [13.005, -4.005], [14.205, -1.305], [12.105, 7.895], [32.205, 2.995], [30.105, 7.595],
                            [25.505, 5.995], [12.705, 6.695], [-33.695, -0.105], [-68.195, 2.695], [-59.995, 4.195],
                            [-56.895, 8.195], [-40.995, 0.895], [-21.495, 5.695], [-6.595, 1.195], [-10.495, -2.205]])
        np.testing.assert_allclose(actual, desired)

    def test_get_parameter_str(self):
        actual = self.dc_offset_remover.get_parameter_str()
        desired = 'No parameters'
        self.assertEqual(actual, desired)
