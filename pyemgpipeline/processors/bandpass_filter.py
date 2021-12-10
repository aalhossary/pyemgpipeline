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

    bf_cutoff_fq_lo : float, default=10
        Low cutoff frequency of the bandpass filter (i.e., frequency
        higher than bf_cutoff_fq_lo will pass).
        See Notes and References for some suggestions in different
        circumstances.

    bf_cutoff_fq_hi : float, default=450
        High cutoff frequency of the bandpass filter (i.e., frequency
        lower than bf_cutoff_fq_hi will pass).
        See Notes and References for some suggestions in different
        circumstances. Note that bf_cutoff_fq_hi should be strictly
        less than half of hz.

    Notes
    -----
    Ref [1] suggests a bandwidth of 10–350 Hz for surface recording,
    10–450 Hz for intramuscular recording, and 10–1500 Hz for needle
    recording.

    Ref [2] suggests a bandwidth of from 5-10 Hz to 400-450 Hz for
    surface EMG and a high cutoff of at least 1500 Hz for
    intramuscular and needle recordings.

    Ref [3] suggests a bandwidth of from 10-20 Hz to 500-1000 Hz for
    surface EMG.

    Ref [4] suggests a low cutoff of 30 Hz when the placement of
    surface EMG is around the heart on the trunk.

    References
    ----------
    [1] Editors (2018).
        Standards for reporting EMG data.
        Journal of Electromyography and Kinesiology, 42, I-II.
        Doi: 10.1016/S1050-6411(18)30348-1.
    [2] Merletti, R, & di Torino, P. (1999).
        Standards for reporting EMG data.
        Journal of Electromyography and Kinesiology, 9(1), III-IV.
    [3] Stegeman, D.F., & Hermens, H.J. (1996-1999).
        Standards for surface electromyography: the European project
        "Surface EMG for non invasive assessment of muscles (SENIAM)".
        Biomed 2 program of the European Community. European concerted
        action.
    [4] Drake, J.D.M., & Callaghan, J.P. (2006).
        Elimination of electrocardiogram contamination from
        electromyogram signals: An evaluation of currently used removal
        techniques.
        Journal of Electromyography and Kinesiology, 16, 175–187.
    """

    def __init__(self, hz, bf_order=2, bf_cutoff_fq_lo=10, bf_cutoff_fq_hi=450):
        assert hz > 2 * bf_cutoff_fq_hi, 'hz must be greater than 2 * bf_cutoff_fq_hi'

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
