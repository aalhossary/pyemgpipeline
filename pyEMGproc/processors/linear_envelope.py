from . base import BaseProcessor
from scipy import signal


class LinearEnvelope(BaseProcessor):
    """Linear envelope

    Parameters
    ----------
    hz : float
        Sample rate in hertz.
        See class EMGMeasurementCollection for suggested values of hz.

    le_order : int, default=2
        Order of the butterworth filter for linear envelope.

    le_cutoff_fq : float, default=6
        Cutoff frequency of the lowpass filter.
        Suggested value is 1~10. For standing scenario, use 1~2.
        For running scenario, use 10.
    """

    def __init__(self, hz, le_order=2, le_cutoff_fq=6):
        self.hz = hz
        self.le_order = le_order
        self.le_cutoff_fq = le_cutoff_fq

    def apply(self, x, **kwargs):
        """Apply linear envelope

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels),
            where n_samples > (le_order + 1) * 3 (See Notes)
            Signal data to be processed.

        Returns
        -------
        x_processed : ndarray of the same shape as x
            The result of applying linear envelope to x.

        Notes
        -----
        Using scipy.signal.filtfilt requires the length of the input
        vector x greater than the padding length, padlen, which is
        (le_order + 1) * 3 for a lowpass Butterworth filter.
        """

        super().assert_input(x)
        assert x.shape[0] > (self.le_order + 1) * 3, 'first dimension of x must have length > (le_order + 1) * 3'

        b, a = signal.butter(N=self.le_order, Wn=self.le_cutoff_fq,
                             btype='lowpass', analog=False, output='ba', fs=self.hz)
        x_processed = signal.filtfilt(b, a, x, axis=0)
        return x_processed

    def get_parameter_str(self):
        """Get the parameters of the linear envelope in str

        Parameters
        ----------
        No parameters

        Returns
        -------
        params_in_str : str
        """

        params_in_str = f'hz = {self.hz}, le_order = {self.le_order}, le_cutoff_fq = {self.le_cutoff_fq}'
        return params_in_str
