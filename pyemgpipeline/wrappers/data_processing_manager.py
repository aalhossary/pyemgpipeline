from .. processors import *
from .. plots import *
from . emg_measurement_collection import EMGMeasurementCollection
import copy


class DataProcessingManager:
    """High-level, guided processing interface with accepted EMG
    processing conventions

    Parameters
    ----------
    c_raw : EMGMeasurementCollection or None
        Containing data (signal, timestamp, hz, channel names, etc.)
        and plot parameters.
        c_raw accepts data from method set_data_and_params and its
        value won't change when running method process_all.

    c : EMGMeasurementCollection or None
        Containing data (signal, timestamp, hz, channel names, etc.)
        and plot parameters.
        When running method process_all, c gets a fresh copy of raw
        data from c_raw. In this way method process_all can be
        repeatedly executed.

    dc_offset_remover : DCOffsetRemover or None
        A DCOffsetRemover processor.

    bandpass_filter : BandpassFilter or None
        A BandpassFilter processor.

    full_wave_rectifier : FullWaveRectifier or None
        A FullWaveRectifier processor.

    linear_envelope : LinearEnvelope or None
        A LinearEnvelope processor.

    end_frame_cutter : EndFrameCutter or None
        An EndFrameCutter processor.

    amplitude_normalizer : AmplitudeNormalizer or None
        An AmplitudeNormalizer processor.

    segmenter : Segmenter or None
        A Segmenter processor.

    segmenter_all_beg_ts : list or None
        Beginning time of interest for each trial to segment. See
        function apply_segmenter of class EMGMeasurementCollection.

    segmenter_all_end_ts : list or None
        End time of interest for each trial to segment. See
        function apply_segmenter of class EMGMeasurementCollection.
    """

    def __init__(self):
        self.c_raw = None
        self.c = None

        self.dc_offset_remover = None
        self.bandpass_filter = None
        self.full_wave_rectifier = None
        self.linear_envelope = None
        self.end_frame_cutter = None
        self.amplitude_normalizer = None
        self.segmenter = None

        self.segmenter_all_beg_ts = None
        self.segmenter_all_end_ts = None

    def set_data_and_params(self, all_data, hz, all_timestamp=None,
                            trial_names=None, channel_names=None, emg_plot_params=None):
        """Set the data and plot parameters and initialize default processors

        Parameters
        ----------
        all_data : list
            Elements of all_data are signal data of the trials.
            Signal data of each trial should be ndarray of shape
            (n_samples,) or (n_samples, n_channels), where n_samples > 15.
            Dimensions and n_channels (if 2-dim) of all trials should be
            the same.

        hz : float
            Sample rate in hertz.

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
        """

        self.c_raw = EMGMeasurementCollection(
            all_data, hz, all_timestamp=all_timestamp, trial_names=trial_names,
            channel_names=channel_names, emg_plot_params=emg_plot_params)

        # set the default 5 processors in case they are None
        if self.dc_offset_remover is None:
            self.dc_offset_remover = DCOffsetRemover()
        if self.bandpass_filter is None:
            self.bandpass_filter = BandpassFilter(self.c_raw.hz)
        if self.full_wave_rectifier is None:
            self.full_wave_rectifier = FullWaveRectifier()
        if self.linear_envelope is None:
            self.linear_envelope = LinearEnvelope(self.c_raw.hz)
        if self.end_frame_cutter is None:
            self.end_frame_cutter = EndFrameCutter()

    def set_dc_offset_remover(self, dc_offset_remover):
        """Set DC offset remover

        Parameters
        ----------
        dc_offset_remover : DCOffsetRemover
            A DC offset remover.

        Returns
        -------
        None
        """

        self.dc_offset_remover = dc_offset_remover

    def set_bandpass_filter(self, bandpass_filter):
        """Set bandpass filter

        Parameters
        ----------
        bandpass_filter : BandpassFilter
            A bandpass filter.

        Returns
        -------
        None
        """

        self.bandpass_filter = bandpass_filter

    def set_full_wave_rectifier(self, full_wave_rectifier):
        """Set full wave rectifier

        Parameters
        ----------
        full_wave_rectifier : FullWaveRectifier
            A full wave rectifier.

        Returns
        -------
        None
        """

        self.full_wave_rectifier = full_wave_rectifier

    def set_linear_envelope(self, linear_envelope):
        """Set linear envelope

        Parameters
        ----------
        linear_envelope : LinearEnvelope
            A linear envelope.

        Returns
        -------
        None
        """

        self.linear_envelope = linear_envelope

    def set_end_frame_cutter(self, end_frame_cutter):
        """Set end frame cutter

        Parameters
        ----------
        end_frame_cutter : EndFrameCutter
            A end frame cutter.

        Returns
        -------
        None
        """

        self.end_frame_cutter = end_frame_cutter

    def set_amplitude_normalizer(self, amplitude_normalizer):
        """Set amplitude normalizer

        Parameters
        ----------
        amplitude_normalizer : AmplitudeNormalizer
            A amplitude normalizer.

        Returns
        -------
        None
        """

        self.amplitude_normalizer = amplitude_normalizer

    def set_segmenter(self, segmenter, all_beg_ts, all_end_ts):
        """Set segmenter and beginning/end time of interest

        Parameters
        ----------
        segmenter : Segmenter
            A segmenter.

        all_beg_ts : list
            Beginning time of interest for each trial.
            See function apply_segmenter of class EMGMeasurementCollection.

        all_end_ts : list
            End time of interest for each trial.
            See function apply_segmenter of class EMGMeasurementCollection.

        Returns
        -------
        None
        """

        self.segmenter = segmenter
        self.segmenter_all_beg_ts = all_beg_ts
        self.segmenter_all_end_ts = all_end_ts

    def show_current_processes_and_related_params(self):
        """Show current processes and related parameter values

        Returns
        -------
        None
        """

        print('---- Current processes and related parameters ----')
        if self.dc_offset_remover is not None:
            print(f'DC offset remover    : {self.dc_offset_remover.get_param_values_in_str()}')
        if self.bandpass_filter is not None:
            print(f'Bandpass filter      : {self.bandpass_filter.get_param_values_in_str()}')
        if self.full_wave_rectifier is not None:
            print(f'Full wave rectifier  : {self.full_wave_rectifier.get_param_values_in_str()}')
        if self.linear_envelope is not None:
            print(f'Linear envelope      : {self.linear_envelope.get_param_values_in_str()}')
        if self.end_frame_cutter is not None:
            print(f'End frame cutter     : {self.end_frame_cutter.get_param_values_in_str()}')
        if self.amplitude_normalizer is not None:
            print(f'Amplitude normalizer : {self.amplitude_normalizer.get_param_values_in_str()}')
        if self.segmenter is not None:
            print(f'Segmenter            : {self.segmenter.get_param_values_in_str()}')

    def _plot_processing_chain(self, current_step,
                               k_for_plot=None, is_overlapping_trials=False, cycled_colors=None,
                               emg_plot_params=None):
        """Plot current step of the processing chain

        Parameters
        ----------
        current_step : str
            Current step in the processing chain.

        k_for_plot : index of list, or None
            If k_for_plot is None, all trials will be plotted.
            If k_for_plot is not None, it should be an integer between
            0 and len(all_data) - 1. In this case, if
            is_overlapping_trials is False, only trial k_for_plot will
            be plotted.

        is_overlapping_trials : bool, default False
            Whether overlapping trials of the same channel.
            If True, all trials will be plotted and overlapped.

        cycled_colors : list or None, default None
            The colors for plotting overlapped trials data.

        emg_plot_params : EMGPlotParams or None, default None
            See class EMGPlotParams and function emg_plot.

        Returns
        -------
        None
        """
        if is_overlapping_trials:
            plot_emg_overlapping_trials(self.c.all_data, self.c.all_timestamp, self.c.trial_names,
                                        cycled_colors=cycled_colors,
                                        channel_names=self.c.channel_names, main_title=current_step,
                                        emg_plot_params=emg_plot_params)
        else:
            for k in range(len(self.c.all_data)):
                if k_for_plot is not None and k != k_for_plot:
                    continue
                plot_emg(self.c.all_data[k], self.c.all_timestamp[k],
                         channel_names=self.c.channel_names,
                         main_title=f'{current_step} ({self.c.trial_names[k]})',
                         emg_plot_params=emg_plot_params)

    def process_all(self, is_plot_processing_chain=False, k_for_plot=None,
                    is_overlapping_trials=False, cycled_colors=None):
        """Apply current processors to data and plot intermediate results

        Parameters
        ----------
        is_plot_processing_chain : bool
            Whether to plot the intermediate results after each
            processing step.

        k_for_plot : index of list, or None
            If k_for_plot is None, all trials will be plotted.
            If k_for_plot is not None, it should be an integer between
            0 and len(all_data) - 1. In this case, if
            is_overlapping_trials is False, only trial k_for_plot will
            be plotted.

        is_overlapping_trials : bool, default False
            Whether overlapping trials of the same channel.
            If True, all trials will be plotted and overlapped.

        cycled_colors : list or None, default None
            The colors for plotting overlapped trials data.

        Returns
        -------
        c : EMGMeasurementCollection
            An EMGMeasurementCollection instance which saves the
            processed data.
        """

        assert self.c_raw is not None, 'Data and parameters must be set by using function set_data_and_params'

        self.c = copy.deepcopy(self.c_raw)

        if k_for_plot is not None:
            assert k_for_plot in range(len(self.c.all_data)), 'k_for_plot must be an index of all_data (a list)'

        emg_plot_params = copy.deepcopy(self.c.emg_plot_params)
        if is_overlapping_trials:
            # when overlapping trials, the color in line2d_kwargs will become invalid.
            # Colors for all trials can be set by users via cycled_colors.
            if emg_plot_params.line2d_kwargs is not None:
                emg_plot_params.line2d_kwargs.pop('color', None)

        if is_plot_processing_chain:
            self._plot_processing_chain('Original', k_for_plot,
                                        is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.dc_offset_remover is not None:
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.dc_offset_remover.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self._plot_processing_chain('After DC offset remover', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.bandpass_filter is not None:
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.bandpass_filter.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self._plot_processing_chain('After bandpass filter', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.full_wave_rectifier is not None:
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.full_wave_rectifier.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self._plot_processing_chain('After full wave rectifier', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.linear_envelope is not None:
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.linear_envelope.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self._plot_processing_chain('After linear envelope', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.end_frame_cutter is not None:
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.end_frame_cutter.apply(self.c.all_data[k])
                self.c.all_timestamp[k] = self.end_frame_cutter.apply(self.c.all_timestamp[k])

            if is_plot_processing_chain:
                self._plot_processing_chain('After end frame cutter', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.amplitude_normalizer is not None:
            max_amplitude = self.c.find_max_amplitude_of_each_channel_across_trials()
            for k in range(len(self.c.all_data)):
                self.c.all_data[k] = self.amplitude_normalizer.apply(self.c.all_data[k], divisor=max_amplitude)

            if is_plot_processing_chain:
                self._plot_processing_chain('After amplitude normalizer', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        if self.segmenter is not None and self.segmenter_all_beg_ts is not None \
                and self.segmenter_all_end_ts is not None:
            for k in range(len(self.c.all_data)):
                beg_idx, end_idx = BaseProcessor.get_indices_from_timestamp(
                    self.c.all_timestamp[k], self.segmenter_all_beg_ts[k], self.segmenter_all_end_ts[k])
                self.c.all_data[k] = Segmenter().apply(self.c.all_data[k], beg_idx=beg_idx, end_idx=end_idx)
                self.c.all_timestamp[k] = Segmenter().apply(self.c.all_timestamp[k], beg_idx=beg_idx, end_idx=end_idx)

            if is_plot_processing_chain:
                self._plot_processing_chain('After segmenter', k_for_plot,
                                            is_overlapping_trials, cycled_colors, emg_plot_params)

        return self.c
