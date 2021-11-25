import numpy as np
import matplotlib.pyplot as plt


class EMGPlotParams:
    """Parameters for EMG plots

    Parameters
    ----------
    n_rows : int, default None
        Number of rows to plot all channels. See function emg_plot for
        more details when considering together with data.

    n_cols : int, default None
        Number of columns to plot all channels. See function emg_plot
        for more details when considering together with data.

    fig_kwargs : dict, default None
        Parameters to control the figures. They are parameters of
        class matplotlib.figure.Figure, including figsize, dpi,
        facecolor, edgecolor, linewidth, frameon, subplotpars,
        tight_layout, constrained_layout
        (See https://matplotlib.org/stable/api/figure_api.html).
    """

    def __init__(self, n_rows=None, n_cols=None, fig_kwargs=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig_kwargs = fig_kwargs


def _get_default_nrows_ncols(n_channels):
    if n_channels <= 3:
        n_cols = 1
    elif n_channels <= 10:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = int(np.ceil(n_channels / n_cols))
    return n_rows, n_cols


def emg_plot(x, timestamp, channel_names=None, main_title=None, emg_plot_params=None):
    """Plot EMG signals on a created matplotlib figure

    Parameters
    ----------
    x : ndarray of shape (n_samples,) or (n_samples, n_channels)
        Signal data to be processed.

    timestamp : ndarray of shape (n_samples,)
        The timestamp corresponding to the signal.

    channel_names : list of str, or None, default None
        If list, its length should be equal to n_channels.
        Channel names to be shown in plots.

    main_title : str or None
        The main title of the plot.

    emg_plot_params : EMGPlotParams, default None
        See class EMGPlotParams.
        If this is not None, then for its n_rows and n_cols,
        (1) If n_rows and n_cols both have values, n_rows times n_cols
        should be no less than n_channels.
        (2) If exactly one of n_rows and n_cols is None, it will get
        the smallest number to fit in all channels.
        (3) If both n_rows and n_cols are None, default setting will be
        used: use one column when n_channels <= 3; use two columns
        when n_channels <= 10; otherwise use three columns.

    Returns
    -------
    None
    """

    if emg_plot_params is not None:
        n_rows, n_cols, fig_kwargs = emg_plot_params.n_rows, emg_plot_params.n_cols, emg_plot_params.fig_kwargs
    else:
        n_rows, n_cols, fig_kwargs = None, None, None

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_channels = x.shape[1]

    if channel_names is not None:
        assert isinstance(channel_names, list) and len(channel_names) == n_channels,\
            'channel_names must be a list with length n_channels'

    if n_rows is None and n_cols is None:
        n_rows, n_cols = _get_default_nrows_ncols(n_channels)
    elif n_rows is None:
        n_rows = int(np.ceil(n_channels / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(n_channels / n_rows))
    else:
        assert n_rows * n_cols >= n_channels, 'n_rows * n_cols >= n_channels must satisfy'

    if fig_kwargs is None:
        fig_kwargs = {}

    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', **fig_kwargs)
    if main_title is not None:
        fig.suptitle(main_title)
    axs = np.array(axs).flat
    for i in range(n_channels):
        axs[i].plot(timestamp, x[:, i])
        if channel_names is not None:
            axs[i].set_title(channel_names[i])

    plt.show()

