import os
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

import inspect
import datetime
import hashlib

import multiprocessing
import itertools
import random




# see https://github.com/ipython/ipython/issues/11049#issue-306086846
def _streams_init_pool():
    sys.stdout.write(".\b") # need to send *something* over stdout to use ipython's display()
    sys.stdout.flush()

#
# Tiny utility for applying a function using multiprocessing.Pool across a
# cartesian product of input arguments, with progress bar
#
def parallel_apply_func_on_input_combinations(
        fn,
        args_values_list,
        *,
        processes=None,
        shuffle_tasks=True
):
    # Notes: (convert to docstring, TODO)
    #
    # - fn should accept a single argument, a tuple of arguments
    #
    # - shuffle_tasks shuffles the order of execution of all tasks; time
    #   progress reporting might be more reliable than if all the hard tasks are
    #   at the end of the values lists.  A cost is that the iterable
    #   representing the cartesian product of input arguments must be flattened
    #   to a list, possibly using lots of memory if you have many input
    #   arguments with many values
    
    from tqdm.auto import tqdm
    import numpy as np

    list_of_inputs = itertools.product(*args_values_list)
    if shuffle_tasks:
        list_of_inputs = list(list_of_inputs)
        random.shuffle(list_of_inputs) # shuffle in place
    with multiprocessing.Pool(processes=processes, initializer=_streams_init_pool) as pool:
        for _ in tqdm(
                pool.imap_unordered( fn, list_of_inputs ),
                total=int(np.prod( [len(x) for x in args_values_list] ))
        ):
            pass


#
# Utility for timing a chunk of code.
#

class TimeThis:
    def __init__(self, dstore=None, attrname='timethisresult', silent=False):
        self.dstore = dstore
        self.silent = silent
        self.attrname = attrname

    def __enter__(self):
        self.tstart = datetime.datetime.now()

    def __exit__(self, *args, **kwargs):
        self.tend = datetime.datetime.now()
        tr = TimeThisResult(tstart=self.tstart, tend=self.tend)
        if self.dstore is not None:
            if isinstance(self.dstore, dict):
                self.dstore[self.attrname] = tr
            else:
                setattr(self.dstore, self.attrname, tr)
        if not self.silent:
            print(tr.report())

class TimeThisResult:
    def __init__(self, tstart, tend):
        super().__init__()
        self.tstart = tstart
        self.tend = tend
        self.dt = tend-tstart
        self.dts = self.dt/datetime.timedelta(seconds=1)
        self.dtus = self.dt/datetime.timedelta(microseconds=1)
    
    def report(self):
        return "Runtime: {} seconds".format(self.dt)

    def __repr__(self):
        return self.__class__.__name__ + '(dt: {})'.format(self.dt)

