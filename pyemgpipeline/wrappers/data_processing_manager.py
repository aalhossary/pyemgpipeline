from .. processors import *
from . emg_measurement_collection import EMGMeasurementCollection, iter_dict_or_list


class DataProcessingManager:
    """Data processing manager which fixes the correct sequence of
    processing steps

    Parameters
    ----------
    c : EMGMeasurementCollection or None
        Containing data (signal, timestamp, hz, channel names, etc.)
        and plot parameters. See class EMGMeasurementCollection.

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

    segmenter_all_beg_ts : dict, list, or None
        Beginning time of interest for each trial to segment. See
        function apply_segmenter of class EMGMeasurementCollection.

    segmenter_all_end_ts : dict, list, or None
        End time of interest for each trial to segment. See
        function apply_segmenter of class EMGMeasurementCollection.
    """

    def __init__(self):
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

    def set_data_and_params(self, all_data, hz, all_timestamp=None, channel_names=None,
                            all_main_titles=None, emg_plot_params=None):
        """Set the data and plot parameters and initialize default processors

        Parameters
        ----------
        all_data : dict or list
            If dict, keys can be trial names and values are signal data of
            the trials.
            If list, elements are signal data of the trials.
            Signal data of each trial should be ndarray of shape
            (n_samples,) or (n_samples, n_channels), where n_samples > 15.
            Dimensions and n_channels (if 2-dim) of all trials should be
            the same.

        hz : float
            Sample rate in hertz.

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
        See class EMGMeasurementCollection.
        """

        self.c = EMGMeasurementCollection(
            all_data, hz, all_timestamp=all_timestamp, channel_names=channel_names,
            all_main_titles=all_main_titles, emg_plot_params=emg_plot_params)

        # set the default 5 processors
        if self.dc_offset_remover is None:
            self.dc_offset_remover = DCOffsetRemover()
        if self.bandpass_filter is None:
            self.bandpass_filter = BandpassFilter(self.c.hz)
        if self.full_wave_rectifier is None:
            self.full_wave_rectifier = FullWaveRectifier()
        if self.linear_envelope is None:
            self.linear_envelope = LinearEnvelope(self.c.hz)
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

        all_beg_ts : dict or list
            Beginning time of interest for each trial.
            See function apply_segmenter of class EMGMeasurementCollection.

        all_end_ts : dict or list
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

    def process_all(self, is_plot_processing_chain=False, k_for_plot=None):
        """Apply current processors to data and plot intermediate results

        Parameters
        ----------
        is_plot_processing_chain : bool
            Whether to plot the intermediate results after each
            processing step.

        k_for_plot : key of dict, index of list, or None
            If all_data is a dict, k_for_plot is one of its key.
            If all_data is a list, k_for_plot is an integer between 0
            and len(all_data) - 1.
            The trial to be plotted. If None, default is the first one.

        Returns
        -------
        c : EMGMeasurementCollection
            An EMGMeasurementCollection instance which saves the
            processed data.
        """

        assert self.c is not None, 'Data and parameters must be set by using function set_data_and_params'

        if is_plot_processing_chain:
            if k_for_plot is None:
                if isinstance(self.c.all_data, dict):
                    k_for_plot = list(self.c.all_data.keys())[0]
                else:
                    k_for_plot = 0
            self.c.plot(k_for_plot, main_title=f'Original ({self.c.all_main_titles[k_for_plot]})')

        if self.dc_offset_remover is not None:
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.dc_offset_remover.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After DC offset remover ({self.c.all_main_titles[k_for_plot]})')

        if self.bandpass_filter is not None:
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.bandpass_filter.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After bandpass filter ({self.c.all_main_titles[k_for_plot]})')

        if self.full_wave_rectifier is not None:
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.full_wave_rectifier.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After full wave rectifier ({self.c.all_main_titles[k_for_plot]})')

        if self.linear_envelope is not None:
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.linear_envelope.apply(self.c.all_data[k])

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After linear envelope ({self.c.all_main_titles[k_for_plot]})')

        if self.end_frame_cutter is not None:
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.end_frame_cutter.apply(self.c.all_data[k])
                self.c.all_timestamp[k] = self.end_frame_cutter.apply(self.c.all_timestamp[k])

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After end frame cutter ({self.c.all_main_titles[k_for_plot]})')

        if self.amplitude_normalizer is not None:
            max_amplitude = self.c.find_max_amplitude_of_each_channel_across_trials()
            for k in iter_dict_or_list(self.c.all_data):
                self.c.all_data[k] = self.amplitude_normalizer.apply(self.c.all_data[k], divisor=max_amplitude)

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After amplitude normalizer ({self.c.all_main_titles[k_for_plot]})')

        if self.segmenter is not None and self.segmenter_all_beg_ts is not None and self.segmenter_all_end_ts is not None:
            for k in iter_dict_or_list(self.c.all_data):
                beg_idx, end_idx = BaseProcessor.get_indices_from_timestamp(
                    self.c.all_timestamp[k], self.segmenter_all_beg_ts[k], self.segmenter_all_end_ts[k])
                self.c.all_data[k] = Segmenter().apply(self.c.all_data[k], beg_idx=beg_idx, end_idx=end_idx)
                self.c.all_timestamp[k] = Segmenter().apply(self.c.all_timestamp[k], beg_idx=beg_idx, end_idx=end_idx)

            if is_plot_processing_chain:
                self.c.plot(k_for_plot, main_title=f'After segmenter ({self.c.all_main_titles[k_for_plot]})')

        return self.c
