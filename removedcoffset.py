"""
Remove the DC from the EMG row signal.

Usage:
    removedcoffset.py [options] <source_database> [<dest_database>]

Options:
    -h -? --help  Print this help screen
"""

import csv
import sys
from pathlib import Path
from typing import Optional

from docpie import docpie
import numpy as np
from numpy import ndarray

from emgio import read_array, write_array


def remove_dc_offset(inp: ndarray) -> ndarray:
    """Remove the DC from passed in array.

    It works well with both single and two dimensional array."""
    avr = np.average(inp, axis=0)
    return inp - avr


if __name__ == '__main__':
    params = docpie(__doc__, sys.argv, version=0.1)
    input_array: Optional[ndarray]
    output_array: Optional[ndarray]
    # input_array, output_array = None, None
    #  read the input file
    _headers, input_array = read_array(params['<source_database>'], params=params)

    # remove DC offset
    output_array = remove_dc_offset(input_array)

    write_array(output_array, _headers, params)
    print("done.")
    sys.exit(0)
