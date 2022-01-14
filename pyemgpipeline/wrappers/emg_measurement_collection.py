from .. processors import *
from .. plots import *
from . emg_measurement import EMGMeasurement
import numpy as np
import copy


def iter_dict_or_list(data_structure):
    """
    Parameters
    ----------
    data_structure : dict or list

    Returns
    -------
    keys_or_indices :
        If data_structure is a dict, keys_or_indices are the keys.
        If data_structure is a list, keys_or_indices are the indices.
    """

    keys_or_indices = data_structure if isinstance(data_structure, dict) else range(len(data_structure))
    return keys_or_indices


class EMGMeasurementCollection:
    """Wrapper of multiple-trial EMG processing

    Parameters
    ----------
    all_data : dict or list
        If dict, keys can be trial names and values are signal data of
        the trials.
        If list, elements are signal data of the trials.
        Signal data of each trial should be ndarray of shape
        (n_samples,) or (n_samples, n_channels), where n_samples > 15
        (See Notes).
        Dimensions and n_channels (if 2-dim) of all trials should be
        the same.

    hz : float
        Sample rate in hertz.
        Ref [1] suggests that minimum sample rate be at least twice the
        highest cutoff frequency of the bandpass filter (Nyquist
        theorem) and preferably higher.
        Ref [2] suggests a minimum sample rate of 1000 Hz for surface
        EMG.
        For fine wire EMG, the authors suggest a minimum sample rate of
        2000 Hz.

    all_timestamp : dict, list, or None, default None
        If dict or list, all_timestamp should be of the same type as
        all_data.
        If dict, keys should be identical to those of all_data and
        values are ndarray or None.
        If list, its length should be the same as the length of all_data
        and elements are ndarray or None.
        The ndarray in all_timestamp is the actual timestamp of the
        corresponding trial, and it should be in 1-dim and have the same
        length as the first dimension of its corresponding element in
        all_data.

    channel_names : list or None, default None
        If list, elements are str and its length should be equal to
        n_channels.
        Channel names of all trials to be shown in plots.

    all_main_titles : list or None, default None
        The main title in the plot, which is valid only when all_data
        is a list. (If all_data is a dict, its keys will be used as
        main titles.)
        If not None, all_main_titles and all_data should have the same
        length.

    emg_plot_params : EMGPlotParams or None, default None
        See class EMGPlotParams and function emg_plot.

    Notes
    -----
    n_samples has to meet the length condition for using bandpass
    filter. With the default parameter in BandpassFilter, n_samples
    must exceed 15. (See class BandpassFilter, function apply)

    References
    ----------
    [1] Editors (2018).
        Standards for reporting EMG data.
        Journal of Electromyography and Kinesiology, 42, I-II.
        Doi: 10.1016/S1050-6411(18)30348-1.
    [2] Stegeman, D.F., & Hermens, H.J. (1999).
        Standards for surface electromyography: the European project
        "Surface EMG for non invasive assessment of muscles (SENIAM)".
        Biomed 2 program of the European Community. European concerted
        action. 108-112.
    """

    def __init__(self, all_data, hz, all_timestamp=None, channel_names=None,
                 all_main_titles=None, emg_plot_params=None):
        assert isinstance(all_data, dict) or isinstance(all_data, list), 'all_data must be dict or list'
        for k in iter_dict_or_list(all_data):
            BaseProcessor.assert_input(all_data[k])
            assert all_data[k].shape[0] > 15, 'first dimension of ndarray in all_data[k] must have length > 15'
        ndims = [all_data[k].ndim for k in iter_dict_or_list(all_data)]
        assert all([e == ndims[0] for e in ndims]), 'Dimensions of all trials should be identical'
        if ndims[0] == 2:
            nchs = [all_data[k].shape[1] for k in iter_dict_or_list(all_data)]
            assert all([e == nchs[0] for e in nchs]), 'n_channels of all trials should be identical'
        self.all_data = copy.deepcopy(all_data)

        self.hz = hz

        self.all_timestamp = copy.deepcopy(all_timestamp)
        if self.all_timestamp is not None:
            if isinstance(self.all_data, dict):
                assert isinstance(self.all_timestamp, dict), 'all_timestamp must be dict since all_data is a dict'
                assert set(self.all_timestamp.keys()) == set(self.all_data.keys()), \
                    'all_timestamp and all_data must have identical keys'
            else:
                assert isinstance(self.all_timestamp, list), 'all_timestamp must be list since all_data is a list'
                assert len(self.all_timestamp) == len(self.all_data), \
                    'The length of all_timestamp and all_data must be identical'
        else:
            if isinstance(self.all_data, dict):
                self.all_timestamp = dict.fromkeys(self.all_data.keys(), None)
            else:
                self.all_timestamp = [None] * len(self.all_data)
        for k in iter_dict_or_list(self.all_data):
            self.all_timestamp[k] = BaseProcessor.get_timestamp(self.all_data[k].shape, self.all_timestamp[k], self.hz)

        self.channel_names = channel_names

        if isinstance(self.all_data, dict):
            self.all_main_titles = {}
            for k in self.all_data:
                self.all_main_titles[k] = k
        else:
            if all_main_titles is None:
                self.all_main_titles = [None] * len(self.all_data)
            else:
                assert isinstance(all_main_titles, list) and len(all_main_titles) == len(self.all_data),\
                    'all_main_titles should be a list with length len(all_data)'
                self.all_main_titles = all_main_titles

        self.emg_plot_params = emg_plot_params

    def apply_dc_offset_remover(self):
        """Apply DC offset remover to the data

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = DCOffsetRemover().apply(self.all_data[k])

    def apply_bandpass_filter(self, bf_order=4, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=450):
        """Apply bandpass filter to the data

        Parameters
        ----------
        bf_order : int, default=4
            Effective order (i.e., order after two-directional
            filtering) of the butterworth filter. bf_order should be
            a multiple of 2.

        bf_cutoff_fq_lo : float, default=10
            Low cutoff frequency of the bandpass filter.
            See class BandpassFilter.

        bf_cutoff_fq_hi : float, default=450
            High cutoff frequency of the bandpass filter.
            See class BandpassFilter.

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = BandpassFilter(self.hz, bf_order, bf_cutoff_fq_lo, bf_cutoff_fq_hi).apply(
                self.all_data[k])

    def apply_full_wave_rectifier(self):
        """Apply full wave rectifier to the data

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = FullWaveRectifier().apply(self.all_data[k])

    def apply_linear_envelope(self, le_order=4, le_cutoff_fq=6):
        """Apply linear envelope to the data

        Parameters
        ----------
        le_order : int, default=4
            Effective order (i.e., order after two-directional
            filtering) of the butterworth filter for linear envelope.
            le_order should be a multiple of 2.

        le_cutoff_fq : float, default=6
            Cutoff frequency of the lowpass filter.
            See class LinearEnvelope.

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = LinearEnvelope(self.hz, le_order, le_cutoff_fq).apply(self.all_data[k])

    def apply_end_frame_cutter(self, n_end_frames=30):
        """Apply end frame cutter to the data (signal and timestamp)

        Parameters
        ----------
        n_end_frames : int, default=30
            Number of frames to be cut off in both ends of the signal.
            n_end_frames >= 0.

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = EndFrameCutter(n_end_frames).apply(self.all_data[k])
            self.all_timestamp[k] = EndFrameCutter(n_end_frames).apply(self.all_timestamp[k])

    def find_max_amplitude_of_each_channel_across_trials(self):
        """Find max amplitude of each channel across trials

        Returns
        -------
        max_amplitude : ndarray
            Shape (n_channels,).
            Max amplitude of each channel which is found across all trials.
        """

        collect_trial_max = ()
        for k in iter_dict_or_list(self.all_data):
            collect_trial_max += tuple(np.max(np.abs(self.all_data[k]), axis=0).reshape(1, -1))
        max_amplitude = np.max(np.vstack(collect_trial_max), axis=0)
        return max_amplitude

    def apply_amplitude_normalizer(self, max_amplitude):
        """Apply amplitude normalizer to the data

        Parameters
        ----------
        max_amplitude : scalar, list, or ndarray
            One or more positive values.
            If data in all_data is in 1-dim or n_channels is 1, then
            max_amplitude should be one value; otherwise max_amplitude
            should be n_channels values.
            max_amplitude is the value used as divisor in amplitude
            normalization for all trials.

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            self.all_data[k] = AmplitudeNormalizer().apply(self.all_data[k], divisor=max_amplitude)

    def apply_segmenter(self, all_beg_ts, all_end_ts):
        """Apply segmenter to the data (signal and timestamp)

        Parameters
        ----------
        all_beg_ts : dict or list
            all_beg_ts should be of the same type as all_data.
            If dict, keys should be identical to those of all_data
            and values are float.
            If list, elements are float and the length should be
            equal to the length of all_data.
            Beginning time of interest for each trial. For trial k,
            all_beg_ts[k] <= all_timestamp[k][-1].

        all_end_ts : dict or list
            all_end_ts should be of the same type as all_data.
            If dict, keys should be identical to those of all_data
            and values are float.
            If list, elements are float and the length should be
            equal to the length of all_data.
            End time of interest for each trial. For trial k,
            all_end_ts[k] >= all_timestamp[k][0] and
            all_beg_ts[k] < all_end_ts[k].

        Returns
        -------
        None
        """

        if isinstance(self.all_data, dict):
            assert isinstance(all_beg_ts, dict) and isinstance(all_end_ts, dict),\
                'all_beg_ts and all_end_ts must be dict since all_data is a dict'
            assert set(all_beg_ts.keys()) == set(all_end_ts.keys()) == set(self.all_data.keys()), \
                'all_beg_ts, all_end_ts, and all_data must have identical keys'
        else:
            assert isinstance(all_beg_ts, list) and isinstance(all_end_ts, list), \
                'all_beg_ts and all_end_ts must be list since all_data is a list'
            assert len(all_beg_ts) == len(all_end_ts) == len(self.all_data), \
                'The length of all_beg_ts, all_end_ts, and all_data must be identical'

        for k in iter_dict_or_list(self.all_data):
            beg_idx, end_idx = BaseProcessor.get_indices_from_timestamp(
                self.all_timestamp[k], all_beg_ts[k], all_end_ts[k])
            self.all_data[k] = Segmenter().apply(self.all_data[k], beg_idx=beg_idx, end_idx=end_idx)
            self.all_timestamp[k] = Segmenter().apply(self.all_timestamp[k], beg_idx=beg_idx, end_idx=end_idx)

    def __getitem__(self, k):
        """Extract data of trial k

        Parameters
        ----------
        k : key of dict or index of list
            If all_data is a dict, k is one of its key.
            If all_data is a list, k is an integer between 0 and
            len(all_data) - 1.

        Returns
        -------
        m : EMGMeasurement
            An EMGMeasurement instance which possesses the data of
            trial k.
        """

        if isinstance(self.all_data, dict):
            assert k in self.all_data.keys(), 'k must be a key of all_data (a dict)'
        else:
            assert k in range(len(self.all_data)), 'k must be an index of all_data (a list)'

        m = EMGMeasurement(self.all_data[k], self.hz, self.all_timestamp[k],
                           self.channel_names, self.all_main_titles[k], self.emg_plot_params)
        return m

    def plot(self):
        """Plot all trials

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            plot_emg(self.all_data[k], self.all_timestamp[k], channel_names=self.channel_names,
                     main_title=self.all_main_titles[k], emg_plot_params=self.emg_plot_params)

    def export_csv(self, all_csv_path):
        """Export the processing results of all trials to csv

        Parameters
        ----------
        all_csv_path : dict or list
            all_csv_path should be of the same type as all_data.
            If dict, keys should be identical to those of all_data.
            If list, its length should be the same as the length of
            all_data.
            The destination paths to export data of all trials.

        Returns
        -------
        None
        """

        for k in iter_dict_or_list(self.all_data):
            BaseProcessor.export_csv(all_csv_path[k], self.all_data[k],
                                     timestamp=self.all_timestamp[k], channel_names=self.channel_names)
