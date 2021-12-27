# EMG Signal Processing Pipeline

**pyemgpipeline** is an electromyography (EMG) signal processing pipeline package.

The main purpose of this package is to ensure that users adhere to accepted EMG processing conventions.
In particular, this package arranges the EMG processing steps together and into a standard order,
providing users with a convenient tool to work on EMG processing without doing it wrong.

The processing steps included in the package are
DC offset removal, bandpass filtering, full wave rectification, linear envelope,
end frame cutting, amplitude normalization, and segmentation.

## Scope

This package defines the processing pipeline for both surface EMG and
intramuscular EMG but not for high density EMG.
The EMG recording requires that the minimum sample rate be at least twice the
highest cutoff frequency of the bandpass filter based on the Nyquist theorem.

## Overview

In **pyemgpipeline**, class `DataProcessingManager` in module `wrappers` is
designed as the main wrapper for high-level, guided processing,
and users are encouraged to use it to adhere to accepted EMG processing steps.
The other classes, methods, and functions are considered as lower level processing
options.

The package is organized in modules `processors`, `wrappers`, and `plots`.

Module `processors` includes the base class `BaseProcessor` of all signal
processors and seven classes for different processing steps:
`DCOffsetRemover`, `BandpassFilter`, `FullWaveRectifier`, `LinearEnvelope`,
`EndFrameCutter`, `AmplitudeNormalizer`, and `Segmenter`.

Module `wrappers` includes three wrapper classes to facilitate the signal
processing by integrating data and individual processors.
Class `EMGMeasurement` works for data of a single trial,
class `EMGMeasurementCollection` works for data of multiple trials,
and class `DataProcessingManager` is the high-level, guided processing wrapper
which fixes the correct sequence of processing steps.

Module `plots` includes
the function `plot_emg` to plot EMG signals on `matplotlib` figures
and the class `EMGPlotParams` to manage the plot-related parameters.

## Documentation

The [documentation](https://aalhossary.github.io/pyemgpipeline/)
describes how to use this package, including
package installation, quick start, examples explaining the breadth of the package’s functionality,
and API reference.

## Community Guidelines

For contribution, please clone the repository, make changes, and create a pull request.

For reporting any issues, please use
[github issues](https://github.com/aalhossary/pyemgpipeline/issues).

For support, please contact the authors via their emails
or [github issues](https://github.com/aalhossary/pyemgpipeline/issues).

## Citation

If you use this package in your project, please cite this work.


