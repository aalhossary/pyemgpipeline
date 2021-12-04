from unittest import TestCase
import numpy as np
from pyemgpipeline.processors import BaseProcessor


class TestBaseProcessor(TestCase):

    def test_assert_input(self):
        # require: x be an ndarray
        with self.assertRaises(AssertionError):
            x = [20.3, 41.0, 53.9, 63.3]  # a list
            BaseProcessor.assert_input(x)

        # require: x be 1-dim or 2-dim ndarray
        with self.assertRaises(AssertionError):
            x = np.array(20.3)  # x.ndim = 0
            BaseProcessor.assert_input(x)

        # require: x be 1-dim or 2-dim ndarray
        with self.assertRaises(AssertionError):
            x = np.array([[[20.3], [41.0]], [[53.9], [63.3]], [[39.5], [24.9]]])  # x.ndim = 3
            BaseProcessor.assert_input(x)

        # require: x have no nan values
        with self.assertRaises(AssertionError):
            x = np.array([20.3, 41.0, np.nan, 63.3])
            BaseProcessor.assert_input(x)

    def test_get_timestamp(self):
        # case 1: output from timestamp
        x_shape = (5,)
        timestamp = np.array([0.2, 0.5, 0.6, 1.0, 1.2])
        hz = 10
        actual = BaseProcessor.get_timestamp(x_shape, timestamp=timestamp, hz=hz)
        desired = np.array([0.2, 0.5, 0.6, 1.0, 1.2])
        np.testing.assert_allclose(actual, desired)

        # case 2: output from hz
        x_shape = (5,)
        timestamp = None
        hz = 10
        actual = BaseProcessor.get_timestamp(x_shape, timestamp=timestamp, hz=hz)
        desired = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(actual, desired)

        # case 3: output from x_shape
        x_shape = (5,)
        timestamp = None
        hz = None
        actual = BaseProcessor.get_timestamp(x_shape, timestamp=timestamp, hz=hz)
        desired = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose(actual, desired)

    def test_get_timestamp__assertion_raise(self):
        x_shape = (5,)  # x_shape[0] = 5
        hz = 10

        # require: timestamp be a 1-dim ndarray
        with self.assertRaises(AssertionError):
            timestamp = [0.2, 0.5, 0.6, 1.0, 1.2]  # a list
            BaseProcessor.get_timestamp(x_shape, timestamp=timestamp, hz=hz)

        # require: timestamp have length equal to x_shape[0]
        with self.assertRaises(AssertionError):
            timestamp = np.array([0.2, 0.5, 0.6, 1.0])  # len(timestamp) = 4
            BaseProcessor.get_timestamp(x_shape, timestamp=timestamp, hz=hz)

    def test_get_indices_from_timestamp(self):
        timestamp = np.array([0.2, 0.5, 0.6, 1.0, 1.2])

        beg_ts1 = 0.2
        end_ts1 = 1.1
        actual1 = BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts1, end_ts=end_ts1)
        desired1 = (0, 3)
        self.assertEqual(actual1, desired1)

        beg_ts2 = 0
        end_ts2 = 1.2
        actual2 = BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts2, end_ts=end_ts2)
        desired2 = (0, 4)
        self.assertEqual(actual2, desired2)

        beg_ts3 = 0.5
        end_ts3 = 2
        actual3 = BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts3, end_ts=end_ts3)
        desired3 = (1, 4)
        self.assertEqual(actual3, desired3)

    def test_get_indices_from_timestamp__assertion_raise(self):
        timestamp = np.array([0.2, 0.5, 0.6, 1.0, 1.2])

        # require: beg_ts <= timestamp[-1]
        with self.assertRaises(AssertionError):
            beg_ts = 1.3
            end_ts = 2.0
            BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts, end_ts=end_ts)

        # require: end_ts >= timestamp[0]
        with self.assertRaises(AssertionError):
            beg_ts = 0
            end_ts = 0.1
            BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts, end_ts=end_ts)

        # require: beg_ts < end_ts
        with self.assertRaises(AssertionError):
            beg_ts = 0.7
            end_ts = 0.7
            BaseProcessor.get_indices_from_timestamp(timestamp, beg_ts=beg_ts, end_ts=end_ts)
