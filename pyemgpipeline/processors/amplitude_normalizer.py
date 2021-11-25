from . base import BaseProcessor
import numpy as np


class AmplitudeNormalizer(BaseProcessor):
    """Amplitude normalizer

    Parameters
    ----------
    No parameters
    """

    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        """Apply amplitude normalizer

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels)
            Signal data to be processed.

        divisor : one or more numbers in scalar, list, or ndarray
            If x is in 1-dim or n_channels is 1, then divisor should be
            one number; otherwise divisor should be n_channels numbers.

        Returns
        -------
        x_processed : ndarray of the same shape as x
            The result of applying amplitude normalizer to x.
        """

        super().assert_input(x)

        assert 'divisor' in kwargs.keys(), 'Need to provide parameter "divisor" in AmplitudeNormalizer.apply'
        divisor = kwargs['divisor']
        divisor = np.array(divisor).flatten()
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            assert divisor.shape == (1,), 'divisor must be one number'
        else:
            assert divisor.shape == (x.shape[1],), 'divisor must be x.shape[1] numbers'

        x_processed = x / divisor
        return x_processed

    def get_parameter_str(self):
        """Get the parameters of the amplitude normalizer in str

        Parameters
        ----------
        No parameters

        Returns
        -------
        params_in_str : str
        """

        params_in_str = 'No parameters'
        return params_in_str
