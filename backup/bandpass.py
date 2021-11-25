"""
Rectify the signal by taking the absolute value of the signal.

Usage:
    bandpass.py [options] <source_database> [<dest_database>]

Options:
    -h -? --help  Print this help screen
"""

import sys
from typing import Optional

from docpie import docpie
import numpy as np
from numpy import ndarray
from scipy.signal import butter, lfilter

from emgio import read_array, write_array

lo = 30  # 20
hi = 750  # 500


def band_pass(inp: ndarray, sr=200) -> ndarray:
    """Filter the signal.

    It works well with both single and two dimensional array."""
    b, a = butter(N=6, Wn=[2 * lo / sr, 2 * hi / sr], btype='band')
    x = lfilter(b, a, inp)

    avr = np.abs(inp)
    return avr


if __name__ == '__main__':
    params = docpie(__doc__, sys.argv, version=0.1)
    input_array: Optional[ndarray]
    output_array: Optional[ndarray]
    # input_array, output_array = None, None
    #  read the input file
    _headers, input_array = read_array(params['<source_database>'], params=params)

    # process
    output_array = band_pass(input_array)

    write_array(output_array, _headers, params)
    print("done.")
    sys.exit(0)
