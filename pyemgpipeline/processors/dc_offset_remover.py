from . base import BaseProcessor
import numpy as np


class DCOffsetRemover(BaseProcessor):
    """DC offset remover

    Parameters
    ----------
    No parameters
    """

    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        """Apply DC offset remover

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels)
            Signal data to be processed.

        Returns
        -------
        x_processed : ndarray of the same shape as x
            The result of applying DC offset remover to x.
        """

        super().assert_input(x)

        x_processed = x - np.mean(x, axis=0)
        return x_processed

    def get_parameter_str(self):
        """Get the parameters of the DC offset remover in str

        Parameters
        ----------
        No parameters

        Returns
        -------
        params_in_str : str
        """

        params_in_str = 'No parameters'
        return params_in_str
