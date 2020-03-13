"""Extract areas of peaks."""
from os.path import dirname, abspath
import os
import re
import itertools
import logging
from sys import argv

import numpy as np
import pandas as pd

ROOT = dirname(abspath(__file__))
COLUMNS = ('CH4', 'C3H6', 'C3H8', 'C4', 'H2', 'O2',
           'N2', 'CO', 'CO2', 'C2H4', 'C2H6', 'C2H2')
INDEX = ((3, 4), (1, 4), (1, 5), (1, 6), (3, 1), (3, 2),
         (3, 3), (3, 5), (2, 4), (2, 6), (2, 7), (2, 5))
PEAK = 'Peak Table'

log = logging.getLogger(__name__)


def clean_data(fn):
    """Extract areas data from raw txt file."""
    areas = []
    with open(fn, 'r') as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if PEAK in line:
                num_peak = int(re.findall('\d+', lines[i+1])[0])
                log.debug("Find peak at line %d, %s", i+1, str.rstrip(line))
                log.debug("Number peaks: %d", num_peak)
                start = i + 2
                # read peak data as a series
                df = pd.read_csv(fn, skiprows=start,
                                 nrows=num_peak, delimiter='\t')
                areas.append(df.values[:, 4])
    values = np.vstack(tuple(itertools.zip_longest(*areas, fillvalue=0)))
    return np.transpose(values)


def get_data():
    """Index all txt data files at current directory.

    Sort fns with run number
    """
    fns = [fn for fn in os.listdir(ROOT) if fn.endswith('.txt')]
    fns = sorted(fns, key=lambda x: int(x.split('.')[0]))
    return fns


def get_area(fn):
    """Get area data from fn.

    Parameters
    ----------
    fn : txt
        data file, e.g., 1.txt
    """
    global INDEX
    val = clean_data(fn)
    idx = np.array(INDEX) - 1
    # separate x y indexing
    x, y = np.transpose(idx)
    res = val[x, y]
    return res


def process_data():
    """Process all data files at current dir.

    Return all area values
    """
    # get all txt fns
    log.info("Load all data files")
    fns = get_data()
    size = len(fns)
    idx_col = []
    values = []
    wrong_fns = []
    # loop through all txt data fns
    for i, fn in enumerate(fns):
        log.info("(%2d/%2d) process %s", i+1, size, fn)
        try:
            res = get_area(fn)
            values.append(res)
            idx_col.append(fn.split('.')[0])    # use fn name as index col
        except Exception:
            log.warning("Cannot process fn %s", fn)
            wrong_fns.append(fn)
    values = np.array(values, dtype=np.float32)
    return idx_col, values, wrong_fns


def write_fn(idx_col, arr):
    """Write array into excel."""
    df = pd.DataFrame(arr, columns=COLUMNS, index=idx_col)
    df.index.name = 'Runs'
    # columns = [('GC area', col) for col in df.columns]
    # df.columns = pd.MultiIndex.from_tuples(columns)
    # df = df.rename_axis([None, 'Runs'], axis=1)
    df.to_excel('result.xlsx')
    return df


if __name__ == '__main__':
    debug = False
    if len(argv) == 2 and argv[1] == 'debug':
        debug = True

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.DEBUG if debug else logging.INFO
    )
    idx_col, values, wrong_fns = process_data()
    log.info("Write data to result.xlsx")
    df = write_fn(idx_col, values)
    if wrong_fns:
        log.warning("!!!Yixiao, check these files, %s", wrong_fns)
