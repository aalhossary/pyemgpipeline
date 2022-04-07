import numpy as np
import matplotlib.pyplot as plt


class EMGPlotParams:
    """Parameters of EMG plots

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
        tight_layout, constrained_layout, etc.
        (See https://matplotlib.org/stable/api/figure_api.html).

    line2d_kwargs : dict, default None
        Parameters to control the plot. They are certain parameters
        of class matplotlib.lines.Line2D, including linewidth,
        linestyle, color, marker, markersize, fillstyle, and more.
        (See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html).
    """

    def __init__(self, n_rows=None, n_cols=None, fig_kwargs=None, line2d_kwargs=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig_kwargs = fig_kwargs
        self.line2d_kwargs = line2d_kwargs


def _get_default_nrows_ncols(n_channels):
    if n_channels <= 3:
        n_cols = 1
    elif n_channels <= 10:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = int(np.ceil(n_channels / n_cols))
    return n_rows, n_cols


def plot_emg(x, timestamp, channel_names=None, main_title=None, emg_plot_params=None):
    """Plot EMG signals on a created matplotlib figure

    Parameters
    ----------
    x : ndarray
        Shape (n_samples,) or (n_samples, n_channels).
        Signal data to be processed.

    timestamp : ndarray
        Shape (n_samples,).
        The timestamp corresponding to the signal.

    channel_names : list or None, default None
        If list, elements are str and its length should be equal to
        n_channels.
        Channel names to be shown in plots.

    main_title : str or None, default None
        The main title of the plot.

    emg_plot_params : EMGPlotParams or None, default None
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
        n_rows, n_cols, fig_kwargs, line2d_kwargs = \
            emg_plot_params.n_rows, emg_plot_params.n_cols, emg_plot_params.fig_kwargs, emg_plot_params.line2d_kwargs
    else:
        n_rows, n_cols, fig_kwargs, line2d_kwargs = None, None, None, None

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

    if line2d_kwargs is None:
        line2d_kwargs = {}

    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', **fig_kwargs)
    plt.xlabel('Time')
    if main_title is not None:
        fig.suptitle(main_title)
    axs = np.array(axs).flat
    for i in range(n_channels):
        axs[i].plot(timestamp, x[:, i], **line2d_kwargs)
        if channel_names is not None:
            axs[i].set_title(channel_names[i])

    plt.show()


def plot_emg_overlapping_trials(all_data, all_timestamp, trial_indices_for_plot, legend_labels,
                                channel_names=None, main_title=None, emg_plot_params=None,
                                cycled_colors=None, legend_kwargs=None, axes_pos_adjust=None):
    """Plot EMG signals on a created matplotlib figure,
    overlapping trials of the same channel

    Parameters
    ----------
    all_data : list
        Elements of all_data are signal data of the trials.
        Signal data of each trial should be ndarray of shape
        (n_samples,) or (n_samples, n_channels).
        Dimensions and n_channels (if 2-dim) of all trials should be
        the same.

    all_timestamp : list
        The length of all_timestamp should be the same as the length
        of all_data.
        Elements of all_timestamp are ndarray of shape (n_samples,).

    trial_indices_for_plot : list of integer
        A list of selected indices of all_data, indicating which trials
        of the data to be plotted.

    legend_labels : list
        The legend labels identify the trials which are overlapped in
        the plots.

    channel_names : list or None, default None
        If list, elements are str and its length should be equal to
        n_channels.
        Channel names to be shown in plots.

    main_title : str or None, default None
        The main title of the plot.

    emg_plot_params : EMGPlotParams or None, default None
        If this is not None, then for its n_rows and n_cols,
        (1) If n_rows and n_cols both have values, n_rows times n_cols
        should be no less than n_channels.
        (2) If exactly one of n_rows and n_cols is None, it will get
        the smallest number to fit in all channels.
        (3) If both n_rows and n_cols are None, default setting will be
        used: use one column when n_channels <= 3; use two columns
        when n_channels <= 10; otherwise use three columns.

    cycled_colors : list or None, default None
        The colors for plotting overlapped trials data.

    legend_kwargs : dict or None, default None
        Parameters to control the legend display. They are the
        "other parameters" of method matplotlib.axes.Axes.legend,
        including loc, bbox_to_anchor, ncol, prop, fontsize, etc.
        (See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html).

    axes_pos_adjust : 4-tuple or None, default None
        Parameters to adjust the axes position (i.e., plot position)
        when legend is displayed to prevent legend overlaying the plot.
        The 4-tuple represents: [0] shift of left position relative to
        width, [1] shift of bottom position relative to height, [2]
        proportion of width, [3] proportion of height.
        If None, no adjustment is applied, i.e., the value (0, 0, 1, 1)
        is applied.

    Returns
    -------
    None
    """

    if emg_plot_params is not None:
        n_rows, n_cols, fig_kwargs, line2d_kwargs = \
            emg_plot_params.n_rows, emg_plot_params.n_cols, emg_plot_params.fig_kwargs, emg_plot_params.line2d_kwargs
    else:
        n_rows, n_cols, fig_kwargs, line2d_kwargs = None, None, None, None

    n_channels = 0
    for k in range(len(all_data)):
        if all_data[k].ndim == 1:
            all_data[k] = all_data[k].reshape(-1, 1)
        n_channels = all_data[k].shape[1]

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

    if line2d_kwargs is None:
        line2d_kwargs = {}

    if legend_kwargs is None:
        legend_kwargs = {}

    if axes_pos_adjust is not None:
        assert isinstance(axes_pos_adjust, tuple) and len(axes_pos_adjust) == 4,\
            'axes_pos_adjust must be a tuple with length 4 if not None'
    else:
        axes_pos_adjust = (0, 0, 1, 1)

    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', **fig_kwargs)
    plt.xlabel('Time')
    if main_title is not None:
        fig.suptitle(main_title)
    axs = np.array(axs).flat

    for i in range(n_channels):
        if cycled_colors is not None:
            axs[i].set_prop_cycle(color=cycled_colors)

        for k in trial_indices_for_plot:
            axs[i].plot(all_timestamp[k], all_data[k][:, i], **line2d_kwargs)

        box = axs[i].get_position()
        axs[i].set_position([box.x0 + box.width * axes_pos_adjust[0],
                             box.y0 + box.height * axes_pos_adjust[1],
                             box.width * axes_pos_adjust[2],
                             box.height * axes_pos_adjust[3]])
        axs[i].legend(labels=legend_labels, **legend_kwargs)

        if channel_names is not None:
            axs[i].set_title(channel_names[i])

    plt.show()
