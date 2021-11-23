from . base import BaseProcessor


class EndFrameCutter(BaseProcessor):
    """End frame cutter

    Parameters
    ----------
    n_end_frames : int, default=30
        Number of frames to be cut off in both ends of the signal.
    """

    def __init__(self, n_end_frames=30):
        self.n_end_frames = n_end_frames

    def apply(self, x, **kwargs):
        """Apply end frame cutter

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, n_channels),
            where n_samples > 2 * n_end_frame
            Signal data to be processed or timestamp corresponding to
            the signal.

        Returns
        -------
        x_processed : ndarray of the same dimension as x, where the
            first dimension reduces its length from 'n_samples' to
            'n_samples - 2 * n_end_frame'
            The result of applying end frame cutter to x.
        """

        super().assert_input(x)
        assert x.shape[0] > 2 * self.n_end_frames, 'first dimension of x must have length > 2 * n_end_frames'

        x_processed = x[self.n_end_frames:(-self.n_end_frames), ]
        return x_processed

    def get_parameter_str(self):
        """Get the parameters of the end frame cutter in str

        Parameters
        ----------
        No parameters

        Returns
        -------
        params_in_str : str
        """

        params_in_str = f'n_end_frames = {self.n_end_frames}'
        return params_in_str
