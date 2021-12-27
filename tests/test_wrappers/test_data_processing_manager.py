from unittest import TestCase
import numpy as np
from pyemgpipeline.wrappers import DataProcessingManager
from pyemgpipeline.processors import BandpassFilter, LinearEnvelope, EndFrameCutter
from pyemgpipeline.processors import AmplitudeNormalizer, Segmenter


class TestDataProcessingManager(TestCase):
    def setUp(self):
        all_data = [
            np.array([20.3, 41.0, 53.9, 63.3, 39.5, 24.9, 26.1, 24.0, 44.1, 42.0,
                      37.4, 24.6, -21.8, -56.3, -48.1, -45.0, -29.1, -9.6, 5.3, 1.4]),
            np.array([1.1, 2.9, 1.4, -0.2, 4.4, 7.2, 9.9, 19.1, 14.2, 18.8,
                      17.2, 17.9, 11.1, 13.9, 15.4, 19.4, 12.1, 16.9, 12.4, 9.0])
        ]
        hz = 1000
        self.mgr = DataProcessingManager()
        self.mgr.set_data_and_params(all_data, hz=hz)
        self.mgr.set_bandpass_filter(BandpassFilter(hz, bf_order=4, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=450))
        self.mgr.set_linear_envelope(LinearEnvelope(hz, le_order=4, le_cutoff_fq=5))
        self.mgr.set_end_frame_cutter(EndFrameCutter(n_end_frames=3))
        self.mgr.set_amplitude_normalizer(AmplitudeNormalizer())
        all_beg_ts = [0.006, 0.0055]
        all_end_ts = [0.012, 0.0095]
        self.mgr.set_segmenter(Segmenter(), all_beg_ts, all_end_ts)

    def tearDown(self):
        pass

    def test_process_all(self):
        c = self.mgr.process_all(is_plot_processing_chain=False)
        actual = c.all_data
        desired = [
            np.array([0.9705620, 0.9753172, 0.9796631, 0.9835977, 0.9871213, 0.9902362, 0.9929473]),
            np.array([-0.6566767, -0.6513617, -0.6464390, -0.6419096])
        ]
        self.assertEqual(len(actual), len(desired))
        for i in range(len(actual)):
            np.testing.assert_allclose(actual[i], desired[i])
