from .. processors import *
from .. plots import *
import copy


class EMGMeasurement:
    """EMG measurement of one trial

    Parameters
    ----------
    data : ndarray of shape (n_samples,) or (n_samples, n_channels),
        where n_samples > 15 (See Notes)
        Signal data of one trial.

    hz : float
        Sample rate in hertz.
        See class EMGMeasurementCollection for suggested values of hz.

    timestamp : ndarray or None, default None
        The actual timestamp corresponds to the signal. If it is an
        ndarray, it should be in 1-dim and have the same length as the
        first dimension of x.

    channel_names : list of str, or None, default None
        If list, its length should be equal to n_channels.
        Channel names to be shown in the plot.

    main_title : str or None, default None
        The main title in the plot.

    emg_plot_params : EMGPlotParams, default None
        Parameters to control the plot. See class EMGPlotParams and
        function emg_plot.

    Notes
    -----
    n_samples has to meet the length condition for using bandpass
    filter. With the default parameter in BandpassFilter, n_samples
    must exceed 15. (See class BandpassFilter, function apply)
    """

    def __init__(self, data, hz, timestamp=None, channel_names=None, main_title=None, emg_plot_params=None):
        BaseProcessor.assert_input(data)
        assert data.shape[0] > 15, 'first dimension of x must have length > 15'
        self.data = copy.deepcopy(data)
        self.hz = hz
        self.timestamp = copy.deepcopy(timestamp)
        self.timestamp = BaseProcessor.get_timestamp(self.data.shape, self.timestamp, self.hz)
        self.channel_names = channel_names
        self.main_title = main_title
        self.emg_plot_params = emg_plot_params

    def apply_dc_offset_remover(self):
        self.data = DCOffsetRemover().apply(self.data)

    def apply_bandpass_filter(self, bf_order=2, bf_cutoff_fq_lo=20, bf_cutoff_fq_hi=499):
        self.data = BandpassFilter(self.hz, bf_order, bf_cutoff_fq_lo, bf_cutoff_fq_hi).apply(self.data)

    def apply_full_wave_rectifier(self):
        self.data = FullWaveRectifier().apply(self.data)

    def apply_linear_envelope(self, le_order=2, le_cutoff_fq=6):
        self.data = LinearEnvelope(self.hz, le_order, le_cutoff_fq).apply(self.data)

    def apply_end_frame_cutter(self, n_end_frames=30):
        self.data = EndFrameCutter(n_end_frames).apply(self.data)
        self.timestamp = EndFrameCutter(n_end_frames).apply(self.timestamp)

    def apply_amplitude_normalizer(self, max_amplitude):
        """
        Parameters
        ----------
        max_amplitude : one or more numbers in scalar, list, or ndarray
            If data is in 1-dim or n_channels is 1, then max_amplitude
            should be one number; otherwise max_amplitude should be
            n_channels numbers.
            max_amplitude is the value used as divisor in amplitude
            normalization.

        Returns
        -------
        None
        """

        self.data = AmplitudeNormalizer().apply(self.data, divisor=max_amplitude)

    def apply_segmenter(self, beg_ts, end_ts):
        """
        Parameters
        ----------
        beg_ts : float
            Beginning time of interest. beg_ts <= timestamp[-1].

        end_ts : float
            End time of interest. end_ts >= timestamp[0].
            Also beg_ts < end_ts.

        Returns
        -------
        None
        """

        beg_idx, end_idx = BaseProcessor.get_indices_from_timestamp(self.timestamp, beg_ts, end_ts)
        self.data = Segmenter().apply(self.data, beg_idx=beg_idx, end_idx=end_idx)
        self.timestamp = Segmenter().apply(self.timestamp, beg_idx=beg_idx, end_idx=end_idx)

    def plot(self):
        """Plot the data
        """
        emg_plot(self.data, self.timestamp, channel_names=self.channel_names,
                 main_title=self.main_title, emg_plot_params=self.emg_plot_params)

    def export_csv(self, csv_path):
        """Export the processing result to csv

        Parameters
        ----------
        csv_path : str
            The destination path to export data.

        Returns
        -------
        None
        """

        BaseProcessor.export_csv(csv_path, self.data, timestamp=self.timestamp, channel_names=self.channel_names)
