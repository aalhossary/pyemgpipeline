from .. processors import *
from . emg_measurement_collection import EMGMeasurementCollection, iter_dict_or_list


class DataProcessingManager:
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
        self.dc_offset_remover = dc_offset_remover

    def set_bandpass_filter(self, bandpass_filter):
        self.bandpass_filter = bandpass_filter

    def set_full_wave_rectifier(self, full_wave_rectifier):
        self.full_wave_rectifier = full_wave_rectifier

    def set_linear_envelope(self, linear_envelope):
        self.linear_envelope = linear_envelope

    def set_end_frame_cutter(self, end_frame_cutter):
        self.end_frame_cutter = end_frame_cutter

    def set_amplitude_normalizer(self, amplitude_normalizer):
        self.amplitude_normalizer = amplitude_normalizer

    def set_segmenter(self, segmenter, all_beg_ts, all_end_ts):
        self.segmenter = segmenter
        self.segmenter_all_beg_ts = all_beg_ts
        self.segmenter_all_end_ts = all_end_ts

    def show_current_processes_and_related_params(self):
        print('---- Current processes and related parameters ----')
        if self.dc_offset_remover is not None:
            print(f'DC offset remover    : {self.dc_offset_remover.get_parameter_str()}')
        if self.bandpass_filter is not None:
            print(f'Bandpass filter      : {self.bandpass_filter.get_parameter_str()}')
        if self.full_wave_rectifier is not None:
            print(f'Full wave rectifier  : {self.full_wave_rectifier.get_parameter_str()}')
        if self.linear_envelope is not None:
            print(f'Linear envelope      : {self.linear_envelope.get_parameter_str()}')
        if self.end_frame_cutter is not None:
            print(f'End frame cutter     : {self.end_frame_cutter.get_parameter_str()}')
        if self.amplitude_normalizer is not None:
            print(f'Amplitude normalizer : {self.amplitude_normalizer.get_parameter_str()}')
        if self.segmenter is not None:
            print(f'Segmenter            : {self.segmenter.get_parameter_str()}')

    def process_all(self, is_plot_processing_chain=False, k_for_plot=None):
        assert self.c is not None, 'Data and parameters must be set by using function set_data_and_params'

        if is_plot_processing_chain:
            if k_for_plot is None:
                if isinstance(self.c.all_data, dict):
                    k_for_plot = list(self.c.all_data.keys())[0]
                else:
                    k_for_plot = 0
            self.c.plot(k_for_plot, main_title=f'Original ({self.c.all_main_titles[k_for_plot]})')

        for k in iter_dict_or_list(self.c.all_data):
            self.c.all_data[k] = self.dc_offset_remover.apply(self.c.all_data[k])

        if is_plot_processing_chain:
            self.c.plot(k_for_plot, main_title=f'After DC offset remover ({self.c.all_main_titles[k_for_plot]})')

        for k in iter_dict_or_list(self.c.all_data):
            self.c.all_data[k] = self.bandpass_filter.apply(self.c.all_data[k])

        if is_plot_processing_chain:
            self.c.plot(k_for_plot, main_title=f'After bandpass filter ({self.c.all_main_titles[k_for_plot]})')

        for k in iter_dict_or_list(self.c.all_data):
            self.c.all_data[k] = self.full_wave_rectifier.apply(self.c.all_data[k])

        if is_plot_processing_chain:
            self.c.plot(k_for_plot, main_title=f'After full wave rectifier ({self.c.all_main_titles[k_for_plot]})')

        for k in iter_dict_or_list(self.c.all_data):
            self.c.all_data[k] = self.linear_envelope.apply(self.c.all_data[k])

        if is_plot_processing_chain:
            self.c.plot(k_for_plot, main_title=f'After linear envelope ({self.c.all_main_titles[k_for_plot]})')

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
