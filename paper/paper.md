---
title: 'pyemgpipeline: A Python package for electromyography processing'
tags:
  - Python
  - electromyography
  - EMG processing
authors:
  - name: Tsung-Lin Wu
    orcid: 0000-0002-1823-6818
    affiliation: 1
  - name: Amr A. Alhossary
    orcid: 0000-0002-4470-5817
    affiliation: 2
  - name: Todd C. Pataky
    orcid: 0000-0002-8292-7189
    affiliation: 3
  - name: Wei Tech Ang
    orcid: 0000-0002-5778-7719
    affiliation: "1, 2"
  - name: Cyril J. Donnelly^[corresponding author]
    orcid: 0000-0003-2443-5212
    affiliation: 2
affiliations:
  - name: Nanyang Technological University, School of Mechanical & Aerospace Engineering
    index: 1
  - name: Nanyang Technological University, Rehabilitation Research Institute of Singapore
    index: 2
  - name: Kyoto University, Department of Human Health Sciences
    index: 3
date: 17 January 2022
bibliography: paper.bib
---

# Summary

We have developed an electromyography (EMG) signal processing pipeline package called `pyemgpipeline`, which is suitable for both surface EMG and intramuscular EMG processing.  `pyemgpipeline` implements internationally accepted EMG processing conventions and provides a high-level interface for ensuring user adherence to those conventions, in terms of (1) processing parameter values, (2) processing steps, and (3) processing step order.  The international standards are from surface EMG for non-invasive assessment of muscles (SENIAM) [@Stegeman:1999].  The seven processing steps included in the package are DC offset removal, bandpass filtering, full wave rectification, linear envelope, end frame cutting, amplitude normalization, and segmentation.  As in sport tasks particularly, it has been observed that amplitudes greater that 100% maximum voluntary contraction (MVC) can be observed [@Devaprakash:2016].  Therefore, we will be using amplitude normalization recommendations from @Devaprakash:2016 for amplitude normalization – this method guarantees a muscle will not exceed 100% MVC as all EMG trials from the experiment are used to identify maximal muscle activation.  What is not included in the package is time event detection and time normalization as different laboratories are interested in different phases of gait and prefer to use either linear or nonlinear time normalized techniques.

As stated by @Stegeman:1999 although EMG is easy to use it is also easy to abuse.  Researchers have thus tried to standardize the use of EMG regarding implementation, analysis, and reporting [@Stegeman:1999; @Merletti:1999; @JEK:2018], these fine works however have arguably been unsuccessfully adopted by many researchers to date.  Based on the published standards and decades of practical experience, we summarize EMG signal processing conventions into the seven steps mentioned above and implement them within an easy-to-use package so that researchers and clinicians with all levels of experience process their EMG signals using correct conventions.  In addition to the processing steps and their order, the choice of processing/filtering parameters can be changed to match the needs of the research or clinician.  For example, within the bandpass filtering step, proper values of the low and high cutoff frequencies of the Butterworth bandpass filter depend on different use cases.  We carefully set default values in the package, and we also include suggested values under different scenarios in [related API documentation](https://aalhossary.github.io/pyemgpipeline/api/pyemgpipeline.processors.html#bandpassfilter) [@JEK:2018; @Merletti:1999; @Stegeman:1999; @Drake:2006].

After the aforementioned seven processing steps, sometimes it is necessary to perform the time normalization step before further analysis.  Time normalization can be executed by an external package `mwarp1d` [@Pataky:2019].

Within this package, class [DataProcessingManager](https://aalhossary.github.io/pyemgpipeline/api/pyemgpipeline.wrappers.html#dataprocessingmanager) is designed for high-level, guided processing, and users are encouraged to use it to adhere to accepted EMG processing conventions.

The depiction of `pyemgpipeline` data processing flow is shown in \autoref{fig:example}, which includes the original signal and the processed signals of all processing steps.  Please refer to the [full documentation](https://aalhossary.github.io/pyemgpipeline/) and the [source code](https://github.com/aalhossary/pyemgpipeline) for detailed information.

# Statement of need

**Research purpose**: `pyemgpipeline` aims to provide software tools for electromyography (EMG) data processing, while ensuring adherence to internationally accepted EMG processing conventions.

**Problem solved**: `pyemgpipeline` implements internationally accepted EMG processing conventions and provides a high-level interface for ensuring user adherence to those conventions. It facilitates the convenience and correctness of processing EMG data. To our best knowledge, no other package provides tools of EMG processing pipeline to ensure users adhere to accepted conventions in terms of (1) processing parameter values, (2) processing steps, (3) processing step order, and (4) amplitude normalization.

**Target audience**: The target audience is anyone working with surface or intramuscular EMG data such as gait biomechanics, sports science, rehabilitation, and robotics.

# Figures

![Depiction of `pyemgpipeline` data processing flow. (x-axis: seconds, y-axis: amplitude.)
\label{fig:example}](example_fig_in_paper.png){ height=80% }

# References
