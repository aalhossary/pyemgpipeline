from .. processors import *
from .. plots import *
from . emg_measurement import EMGMeasurement
import numpy as np
import copy


class EMGMeasurementCollection:
    """Wrapper of multiple-trial EMG processing

    Parameters
    ----------
    all_data : list
        Elements of all_data are signal data of the trials.
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

    all_timestamp : list or None, default None
        If list, its length should be the same as the length of all_data
        and elements are ndarray or None.
        The ndarray in all_timestamp is the actual timestamp of the
        corresponding trial, and it should be in 1-dim and have the same
        length as the first dimension of its corresponding element in
        all_data.

    trial_names : list or None, default None
        Trial names.
        If not None, trial_names and all_data should have the same
        length.

    channel_names : list or None, default None
        If list, elements are str and its length should be equal to
        n_channels.
        Channel names of all trials to be shown in plots.

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
        "Surface EMG for non-invasive assessment of muscles (SENIAM)".
        Biomed 2 program of the European Community. European concerted
        action. 108-112.
    """

    def __init__(self, all_data, hz, all_timestamp=None,
                 trial_names=None, channel_names=None, emg_plot_params=None):
        assert isinstance(all_data, list), 'all_data must be list'
        for k in range(len(all_data)):
            BaseProcessor.assert_input(all_data[k])
            assert all_data[k].shape[0] > 15, 'first dimension of ndarray in all_data[k] must have length > 15'
        ndims = [all_data[k].ndim for k in range(len(all_data))]
        assert all([e == ndims[0] for e in ndims]), 'Dimensions of all trials should be identical'
        if ndims[0] == 2:
            nchs = [all_data[k].shape[1] for k in range(len(all_data))]
            assert all([e == nchs[0] for e in nchs]), 'n_channels of all trials should be identical'
        self.all_data = copy.deepcopy(all_data)

        self.hz = hz

        self.all_timestamp = copy.deepcopy(all_timestamp)
        if self.all_timestamp is not None:
            assert isinstance(self.all_timestamp, list), 'all_timestamp must be list if not None'
            assert len(self.all_timestamp) == len(self.all_data), \
                'The length of all_timestamp and all_data must be identical'
        else:
            self.all_timestamp = [None] * len(self.all_data)
        for k in range(len(all_data)):
            self.all_timestamp[k] = BaseProcessor.get_timestamp(self.all_data[k].shape, self.all_timestamp[k], self.hz)

        if trial_names is None:
            self.trial_names = [None] * len(self.all_data)
        else:
            assert isinstance(trial_names, list) and len(trial_names) == len(self.all_data), \
                'trial_names should be a list with length len(all_data)'
            self.trial_names = trial_names

        self.channel_names = channel_names

        self.emg_plot_params = emg_plot_params

    def apply_dc_offset_remover(self):
        """Apply DC offset remover to the data

        Returns
        -------
        None
        """

        for k in range(len(self.all_data)):
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

        for k in range(len(self.all_data)):
            self.all_data[k] = BandpassFilter(self.hz, bf_order, bf_cutoff_fq_lo, bf_cutoff_fq_hi).apply(
                self.all_data[k])

    def apply_full_wave_rectifier(self):
        """Apply full wave rectifier to the data

        Returns
        -------
        None
        """

        for k in range(len(self.all_data)):
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

        for k in range(len(self.all_data)):
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

        for k in range(len(self.all_data)):
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
        for k in range(len(self.all_data)):
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

        for k in range(len(self.all_data)):
            self.all_data[k] = AmplitudeNormalizer().apply(self.all_data[k], divisor=max_amplitude)

    def apply_segmenter(self, all_beg_ts, all_end_ts):
        """Apply segmenter to the data (signal and timestamp)

        Parameters
        ----------
        all_beg_ts : list
            Elements of all_beg_ts are float and the length should be
            equal to the length of all_data.
            Beginning time of interest for each trial. For trial k,
            all_beg_ts[k] <= all_timestamp[k][-1].

        all_end_ts : list
            Elements of all_end_ts are float and the length should be
            equal to the length of all_data.
            End time of interest for each trial. For trial k,
            all_end_ts[k] >= all_timestamp[k][0] and
            all_beg_ts[k] < all_end_ts[k].

        Returns
        -------
        None
        """

        assert isinstance(all_beg_ts, list) and isinstance(all_end_ts, list), \
            'all_beg_ts and all_end_ts must be list since all_data is a list'
        assert len(all_beg_ts) == len(all_end_ts) == len(self.all_data), \
            'The length of all_beg_ts, all_end_ts, and all_data must be identical'

        for k in range(len(self.all_data)):
            beg_idx, end_idx = BaseProcessor.get_indices_from_timestamp(
                self.all_timestamp[k], all_beg_ts[k], all_end_ts[k])
            self.all_data[k] = Segmenter().apply(self.all_data[k], beg_idx=beg_idx, end_idx=end_idx)
            self.all_timestamp[k] = Segmenter().apply(self.all_timestamp[k], beg_idx=beg_idx, end_idx=end_idx)

    def __getitem__(self, k):
        """Extract data of trial k

        Parameters
        ----------
        k : index of list
            k is an integer between 0 and len(all_data) - 1.

        Returns
        -------
        m : EMGMeasurement
            An EMGMeasurement instance which possesses the data of
            trial k.
        """

        assert k in range(len(self.all_data)), 'k must be an index of all_data (a list)'

        m = EMGMeasurement(self.all_data[k], self.hz, self.all_timestamp[k],
                           self.trial_names[k], self.channel_names, self.emg_plot_params)
        return m

    def plot(self, k_for_plot=None, is_overlapping_trials=False, main_title=None,
             cycled_colors=None, is_hide_legend=False, legend_kwargs=None, axes_pos_adjust=None):
        """Plot all trials

        Parameters
        ----------
        k_for_plot : integer, list of integer, or None
            If integer, k_for_plot is an index of the list
            self.c.all_data.
            If list of integer, k_for_plot is a list of selected
            indices of the list self.c.all_data.
            If None, k_for_plot will be a list of all indices of the
            list self.c.all_data.
            k_for_plot sets which trials of the data to be plotted.

        is_overlapping_trials : bool, default False
            Whether or not to plot trials of the same channel
            overlappingly on one (sub)figure.

        main_title : str or None, default None
            main_title is used when is_overlapping_trials is True.
            The main title of the plot.

        cycled_colors : list or None, default None
            cycled_colors is used when is_overlapping_trials is True.
            The colors for plotting overlapped trials data.

        is_hide_legend : bool, default False
            Whether or not to hide legend.

        legend_kwargs : dict or None, default None
            legend_kwargs is used when is_overlapping_trials is True.
            Parameters to control the legend display. They are the
            "other parameters" of method matplotlib.axes.Axes.legend,
            including loc, bbox_to_anchor, ncol, prop, fontsize, etc.
            (See
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html).

        axes_pos_adjust : 4-tuple or None, default None
            axes_pos_adjust is used when is_overlapping_trials is True.
            Parameters to adjust the axes position (i.e., plot position)
            when legend is displayed to prevent legend overlaying the plot.
            The 4-tuple represents: [0] shift of left position relative to
            width, [1] shift of bottom position relative to height, [2]
            proportion of width, [3] proportion of height.
            If None, no adjustment is applied, i.e., the value (0, 0, 1, 1)
            is applied.

        Returns
        -------
        None
        """

        if k_for_plot is not None:
            trial_indices_for_plot = k_for_plot
            if not isinstance(trial_indices_for_plot, list):
                trial_indices_for_plot = [trial_indices_for_plot]  # Now this is a list
            for k in trial_indices_for_plot:
                assert k in range(len(self.all_data)), 'Value(s) in k_for_plot must be indices of the list all_data'
        else:
            trial_indices_for_plot = list(range(len(self.all_data)))  # Now this is a list of all indices all_data

        if is_overlapping_trials:
            # when overlapping trials, the color in line2d_kwargs will become invalid.
            # Colors for all trials can be set by users via cycled_colors.
            emg_plot_params = copy.deepcopy(self.emg_plot_params)
            if emg_plot_params.line2d_kwargs is not None:
                emg_plot_params.line2d_kwargs.pop('color', None)

            legend_labels = [self.trial_names[k] for k in trial_indices_for_plot]
            plot_emg_overlapping_trials(self.all_data, self.all_timestamp,
                                        trial_indices_for_plot, legend_labels,
                                        channel_names=self.channel_names, main_title=main_title,
                                        emg_plot_params=emg_plot_params, cycled_colors=cycled_colors,
                                        is_hide_legend=is_hide_legend, legend_kwargs=legend_kwargs,
                                        axes_pos_adjust=axes_pos_adjust)
        else:
            for k in trial_indices_for_plot:
                plot_emg(self.all_data[k], self.all_timestamp[k], channel_names=self.channel_names,
                         main_title=self.trial_names[k], emg_plot_params=self.emg_plot_params)

    def export_csv(self, all_csv_path):
        """Export the processing results of all trials to csv

        Parameters
        ----------
        all_csv_path : list
            The length of all_csv_path should be the same as the length
            of all_data.
            The destination paths to export data of all trials.

        Returns
        -------
        None
        """

        for k in range(len(self.all_data)):
            BaseProcessor.export_csv(all_csv_path[k], self.all_data[k],
                                     timestamp=self.all_timestamp[k], channel_names=self.channel_names)
