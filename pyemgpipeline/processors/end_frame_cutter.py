from . base import BaseProcessor


class EndFrameCutter(BaseProcessor):
    """End frame cutter for EMG signals

    Parameters
    ----------
    n_end_frames : int, default=30
        Number of frames to be cut off in both ends of the signal.
        n_end_frames >= 0.
    """

    def __init__(self, n_end_frames=30):
        assert n_end_frames >= 0, 'n_end_frames must be non-negative'
        self.n_end_frames = n_end_frames

    def apply(self, x, **kwargs):
        """Apply end frame cutter

        Parameters
        ----------
        x : ndarray
            Shape (n_samples,) or (n_samples, n_channels),
            where n_samples > 2 * n_end_frame.
            Signal data to be processed or timestamp corresponding to
            the signal.

        Returns
        -------
        x_processed : ndarray
            Same dimension as x, where the first dimension reduces its
            length from 'n_samples' to 'n_samples - 2 * n_end_frame'.
            The result of applying end frame cutter to x.
        """

        super().assert_input(x)
        assert x.shape[0] > 2 * self.n_end_frames, 'first dimension of x must have length > 2 * n_end_frames'

        if self.n_end_frames > 0:
            x_processed = x[self.n_end_frames:(-self.n_end_frames), ]
        else:
            x_processed = x
        return x_processed

    def get_param_values_in_str(self):
        """Getting the parameter values of the processor for display
        purpose

        Returns
        -------
        params_in_str : str
            Parameter values.
        """

        params_in_str = f'n_end_frames = {self.n_end_frames}'
        return params_in_str
