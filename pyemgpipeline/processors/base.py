from abc import ABCMeta, abstractmethod
import numpy as np


class BaseProcessor(metaclass=ABCMeta):
    """Base class for all signal processors in pyemgpipeline
    """

    @abstractmethod
    def apply(self, x, **kwargs):
        """Abstract method of applying any EMG signal processing steps

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels),
            where processors may require conditions on n_samples.
            Signal data to be processed.

        **kwargs : processor-specific arguments

        Returns
        -------
        None (in abstract method)
        """

        return

    @abstractmethod
    def get_parameter_str(self):
        """Abstract method of getting the parameters of the processor
        for display purpose

        Parameters
        ----------
        No parameters

        Returns
        -------
        Parameters of the processor in str
        """

        return ""

    @staticmethod
    def assert_input(x):
        """Check the input signal

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels)
            Signal data to be processed.

        Returns
        -------
        None
        """

        assert isinstance(x, np.ndarray), 'x must be an ndarray'
        assert x.ndim == 1 or x.ndim == 2, 'x must be a 1-dim or 2-dim ndarray'
        assert not np.isnan(np.sum(x)), 'x must not contain nan values'

    @staticmethod
    def get_timestamp(x_shape, timestamp=None, hz=None):
        """Get the timestamp from one of the arguments

        Parameters
        ----------
        x_shape : shape of the signal data in ndarray
            x_shape is (n_samples,) or (n_samples, n_channels).

        timestamp : ndarray or None, default None
            The actual timestamp corresponding to the signal. If it is
            an ndarray, it should be in 1-dim and have length equal to
            x_shape[0].

        hz : float or None, default None
            Sample rate in hertz.

        Returns
        -------
        timestamp: ndarray of shape (n_samples,)
            The deduced timestamp corresponding to the signal. If
            getting None for both timestamp and hz from the input,
            this return value will be integers starting from 0.
        """

        if timestamp is not None:
            assert isinstance(timestamp, np.ndarray), 'timestamp must be an ndarray'
            assert timestamp.ndim == 1 and timestamp.shape[0] == x_shape[0], \
                'timestamp must be in 1-dim and have length equal to x_shape[0]'
            assert not np.isnan(np.sum(timestamp)), 'timestamp must not contain nan values'
        elif hz is not None:
            assert isinstance(hz, float) or isinstance(hz, int), 'hz must be a number'
            timestamp = np.array([e / hz for e in range(x_shape[0])])
        else:
            timestamp = np.array(range(x_shape[0]))
        return timestamp

    @staticmethod
    def get_indices_from_timestamp(timestamp, beg_ts, end_ts):
        """Get the corresponding indices from timestamp

        Parameters
        ----------
        timestamp : ndarray in 1-dim

        beg_ts : float
            Beginning time of interest. beg_ts <= timestamp[-1].

        end_ts : float
            End time of interest. end_ts >= timestamp[0].
            Also beg_ts < end_ts.

        Returns
        -------
        beg_idx : int
            The smallest index of timestamp in which the time is
            greater than or equal to beg_ts.

        end_idx : int
            The largest index of timestamp in which the time is
            less than or equal to end_ts.
        """

        assert beg_ts < end_ts, 'beg_ts must be less than end_ts'
        assert beg_ts <= timestamp[-1] and end_ts >= timestamp[0],\
            'beg_ts must be no greater than timestamp[-1] and end_ts must be no less than timestamp[0]'

        beg_idx = int(np.where(timestamp >= beg_ts)[0][0])
        end_idx = int(np.where(timestamp <= end_ts)[0][-1])
        return beg_idx, end_idx

    @staticmethod
    def export_csv(filepath, x, timestamp=None, channel_names=None):
        """Export data to csv file

        Parameters
        ----------
        filepath : str
            Filepath for data to export.

        x : ndarray of shape (n_samples,) or (n_samples, n_channels)
            Signal data to be exported.

        timestamp : ndarray of shape (n_samples,) or None, default None
            The timestamp corresponding to the signal.

        channel_names : list of str, or None, default None
            If list, its length should be equal to n_channels.
            Channel names to be shown as the header of the csv file.

        Returns
        -------
        None
        """

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_channels = x.shape[1]

        if timestamp is not None:
            out_data = np.hstack((timestamp.reshape(-1,1), x))
        else:
            out_data = x

        if channel_names is not None:
            assert isinstance(channel_names, list) and len(channel_names) == n_channels, \
                'channel_names must be a list with length n_channels'
            header = ','.join(channel_names)
            if timestamp is not None:
                header = 'Timestamp,' + header
        else:
            header = ''

        np.savetxt(filepath, out_data, delimiter=',', header=header, comments='')
