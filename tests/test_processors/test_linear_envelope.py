from unittest import TestCase
import numpy as np
from pyemgpipeline.processors import LinearEnvelope


class TestLinearEnvelope(TestCase):
    def setUp(self):
        hz = 1000
        le_order = 4
        le_cutoff_fq = 5
        self.linear_envelope = LinearEnvelope(hz=hz, le_order=le_order, le_cutoff_fq=le_cutoff_fq)

    def tearDown(self):
        pass

    def test_apply__case_x_1dim(self):
        x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4])
        actual = self.linear_envelope.apply(x)
        desired = np.array([3.2664113, 3.3081268, 3.3472161, 3.3835973, 3.4172337,
                            3.4481353, 3.4763497, 3.5019513, 3.5250378, 3.5457335,
                            3.5641951, 3.5806142, 3.5952102, 3.6082107, 3.6198237,
                            3.6302147, 3.6395004, 3.6477546, 3.6550211, 3.6613283])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_1channel(self):
        x = np.array([[20.3], [41.0], [53.9], [63.3], [39.5], [24.9], [26.1], [24.0], [44.1], [42.0],
                      [37.4], [24.6], [-21.8], [-56.3], [-48.1], [-45.0], [-29.1], [-9.6], [5.3], [1.4]])
        actual = self.linear_envelope.apply(x)
        desired = np.array([[3.2664113], [3.3081268], [3.3472161], [3.3835973], [3.4172337],
                            [3.4481353], [3.4763497], [3.5019513], [3.5250378], [3.5457335],
                            [3.5641951], [3.5806142], [3.5952102], [3.6082107], [3.6198237],
                            [3.6302147], [3.6395004], [3.6477546], [3.6550211], [3.6613283]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__case_x_2dim_2channel(self):
        x = np.array([[20.3, 1.1], [41.0, 2.9], [53.9, 1.4], [63.3, -0.2], [39.5, 4.4],
                      [24.9, 7.2], [26.1, 9.9], [24.0, 19.1], [44.1, 14.2], [42.0, 18.8],
                      [37.4, 17.2], [24.6, 17.9], [-21.8, 11.1], [-56.3, 13.9], [-48.1, 15.4],
                      [-45.0, 19.4], [-29.1, 12.1], [-9.6, 16.9], [5.3, 12.4], [1.4, 9.0]])
        actual = self.linear_envelope.apply(x)
        desired = np.array([[3.2664113, -9.4276525], [3.3081268, -9.3487550], [3.3472161, -9.2732103],
                            [3.3835973, -9.2011398], [3.4172337, -9.1326544], [3.4481353, -9.0678542],
                            [3.4763497, -9.0068260], [3.5019513, -8.9496410], [3.5250378, -8.8963502],
                            [3.5457335, -8.8469814], [3.5641951, -8.8015377], [3.5806142, -8.7599969],
                            [3.5952102, -8.7223115], [3.6082107, -8.6884097], [3.6198237, -8.6581978],
                            [3.6302147, -8.6315597], [3.6395004, -8.6083555], [3.6477546, -8.5884216],
                            [3.6550211, -8.5715707], [3.6613283, -8.5575935]])
        np.testing.assert_allclose(actual, desired)

    def test_apply__assertion_raise(self):
        # require: n_samples > (le_order / 2 + 1) * 3
        with self.assertRaises(AssertionError):
            x = np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1])  # n_samples = 9
            self.linear_envelope.apply(x)  # (le_order / 2 + 1) * 3 = 9

    def test_get_parameter_str(self):
        actual = self.linear_envelope.get_param_values_in_str()
        desired = 'hz = 1000, le_order = 4, le_cutoff_fq = 5'
        self.assertEqual(actual, desired)
