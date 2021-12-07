from . base import BaseProcessor
from scipy import signal


class BandpassFilter(BaseProcessor):
    """Bandpass filter for EMG signals

    Parameters
    ----------
    hz : float
        Sample rate in hertz.
        See class EMGMeasurementCollection for suggested values of hz.

    bf_order : int, default=2
        Order of the butterworth filter.

    bf_cutoff_fq_lo : float, default=20
        Low cutoff frequency of the bandpass filter (i.e., frequency
        higher than bf_cutoff_fq_lo will pass).
        For EMG placement on lower/upper limbs, the suggested value is
        10~20 (see ref. 1). For EMG placement around the heart (i.e.,
        trunk), the suggested value is 30 (see ref. 2).

    bf_cutoff_fq_hi : float, default=499
        High cutoff frequency of the bandpass filter (i.e., frequency
        lower than bf_cutoff_fq_hi will pass).
        Suggested value is around 500 (see ref. 1). Note that
        bf_cutoff_fq_hi should be strictly less than half of hz.

    References
    ----------
    1. Stegeman, D.F., & Hermens, H.J. (1996-1999).
        Standards for surface electromyography: the European project
        "Surface EMG for non invasive assessment of muscles (SENIAM)".
        Biomed 2 program of the European Community. European concerted
        action.
    2. Drake, J.D.M., & Callaghan, J.P. (2006).
        Elimination of electrocardiogram contamination from
        electromyogram signals: An evaluation of currently used removal
        techniques. Journal of Electromyography and Kinesiology, 16,
        175–187.
    """

    def __init__(self, hz, bf_order=2, bf_cutoff_fq_lo=20, bf_cutoff_fq_hi=499):
        self.hz = hz
        self.bf_order = bf_order
        self.bf_cutoff_fq_lo = bf_cutoff_fq_lo
        self.bf_cutoff_fq_hi = bf_cutoff_fq_hi

    def apply(self, x, **kwargs):
        """Apply bandpass filter

        Parameters
        ----------
        x : ndarray
            Shape (n_samples,) or (n_samples, n_channels),
            where n_samples > (2 * bf_order + 1) * 3 (See Notes).
            Signal data to be processed.

        Returns
        -------
        x_processed : ndarray
            Same shape as x.
            The result of applying bandpass filter to x.

        Notes
        -----
        Using scipy.signal.filtfilt requires the length of the input
        vector x greater than the padding length, padlen, which is
        (2 * bf_order + 1) * 3 for a bandpass Butterworth filter.
        """

        super().assert_input(x)
        assert x.shape[0] > (2 * self.bf_order + 1) * 3, 'first dimension of x must have length > (2 * bf_order + 1) * 3'

        b, a = signal.butter(N=self.bf_order, Wn=[self.bf_cutoff_fq_lo, self.bf_cutoff_fq_hi],
                             btype='bandpass', analog=False, output='ba', fs=self.hz)
        x_processed = signal.filtfilt(b, a, x, axis=0)
        return x_processed

    def get_param_values_in_str(self):
        """Getting the parameter values of the processor for display
        purpose

        Returns
        -------
        params_in_str : str
            Parameter values.
        """

        params_in_str = f'hz = {self.hz}, bf_order = {self.bf_order}, bf_cutoff_fq_lo = {self.bf_cutoff_fq_lo}, ' \
                        f'bf_cutoff_fq_hi = {self.bf_cutoff_fq_hi}'
        return params_in_str
