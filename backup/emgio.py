import csv
import sys
from pathlib import Path
from typing import Dict, Tuple, Iterable

import numpy as np
from numpy import ndarray


def read_array(src_str: str, params: Dict) -> Tuple[Iterable[str], ndarray]:
    source_file = Path(params['<source_database>'])
    with open(source_file, 'r') as sf:
        # input_array = np.genfromtxt(sf, dtype=float, delimiter=',', skip_header=1)
        reader = csv.reader(sf, delimiter=',')
        _headers = next(reader)
        input_array = np.array(list(reader)).astype(float)
        return _headers, input_array


def write_array(output_array: ndarray, headers: Iterable[str], params: Dict):
    if params.get('<dest_database>', None):
        dest_file = Path(params['<dest_database>'])
        outfile = open(dest_file, 'w')
    else:
        outfile = sys.stdout
    # write the output file
    print(', '.join(headers), file=outfile)
    np.savetxt(outfile, output_array, fmt='%.9f', delimiter=',')
