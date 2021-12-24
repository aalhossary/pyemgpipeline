from . base import BaseProcessor
import numpy as np


class Segmenter(BaseProcessor):
    """Segmenter for EMG signals

    """

    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        """Apply segmenter

        Parameters
        ----------
        x : ndarray
            Shape (n_samples,) or (n_samples, n_channels).
            Signal data to be processed or timestamp corresponding to
            the signal.

        beg_idx : int
            Beginning index (inclusive) for segmentation of x at its
            first dimension. 0 <= beg_idx <= end_idx < n_samples.

        end_idx : int
            End index (inclusive) for segmentation of x at its first
            dimension. 0 <= beg_idx <= end_idx < n_samples.

        Returns
        -------
        x_processed : ndarray
            Same dimension as x, where the first dimension reduces its
            length from 'n_samples' to 'end_idx - beg_idx + 1'.
            The result of applying segmenter to x.
        """

        super().assert_input(x)

        assert 'beg_idx' in kwargs.keys(), 'Need to provide parameter "beg_idx" in Segmenter.apply'
        assert 'end_idx' in kwargs.keys(), 'Need to provide parameter "end_idx" in Segmenter.apply'
        beg_idx = kwargs['beg_idx']
        end_idx = kwargs['end_idx']
        assert isinstance(beg_idx, (int, np.integer)) and isinstance(end_idx, (int, np.integer))\
               and 0 <= beg_idx <= end_idx < x.shape[0],\
               'beg_idx and end_idx must be integers and 0 <= beg_idx <= end_idx < x.shape[0]'

        x_processed = x[beg_idx:(end_idx + 1), ]
        return x_processed

    def get_param_values_in_str(self):
        """Getting the parameter values of the processor for display
        purpose

        Returns
        -------
        params_in_str : str
            Parameter values.
        """

        params_in_str = super().get_param_values_in_str()
        return params_in_str
