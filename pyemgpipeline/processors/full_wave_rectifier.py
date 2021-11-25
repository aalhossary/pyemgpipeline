from . base import BaseProcessor
import numpy as np


class FullWaveRectifier(BaseProcessor):
    """Full wave rectifier

    Parameters
    ----------
    No parameters
    """

    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        """Apply full wave rectifier

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels)
            Signal data to be processed.

        Returns
        -------
        x_processed : ndarray of the same shape as x
            The result of applying full wave rectifier to x.
        """

        super().assert_input(x)

        x_processed = np.abs(x)
        return x_processed

    def get_parameter_str(self):
        """Get the parameters of the full wave rectifier in str

        Parameters
        ----------
        No parameters

        Returns
        -------
        params_in_str : str
        """

        params_in_str = 'No parameters'
        return params_in_str
