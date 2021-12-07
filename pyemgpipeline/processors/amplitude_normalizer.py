from . base import BaseProcessor
import numpy as np


class AmplitudeNormalizer(BaseProcessor):
    """Amplitude normalizer for EMG signals

    """

    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        """Apply amplitude normalizer

        Parameters
        ----------
        x : ndarray
            Shape (n_samples,) or (n_samples, n_channels).
            Signal data to be processed.

        divisor : scalar, list, or ndarray
            One or more positive values.
            If x is in 1-dim or n_channels is 1, then divisor should be
            one value; otherwise divisor should be n_channels values.

        Returns
        -------
        x_processed : ndarray
            Same shape as x.
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

        assert np.all(divisor > 0), 'divisor must contain positive values'

        x_processed = x / divisor
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
