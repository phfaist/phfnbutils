import os
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

import inspect
import datetime
#import hashlib

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
        *args_values_lists,
        #
        processes=None,
        shuffle_tasks=True,
        sequential_execution=False,
        chunksize=None,
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

    args_values_lists = list(args_values_lists)

    list_of_inputs = itertools.chain.from_iterable(
        (itertools.product(*args_values_list) for args_values_list in args_values_lists)
    )
    if shuffle_tasks:
        list_of_inputs = list(list_of_inputs)
        random.shuffle(list_of_inputs) # shuffle in place
        total_num_inputs = len(list_of_inputs)
    else:
        total_num_inputs = sum(
            int(np.prod( [len(x) for x in args_values_list] ))
            for args_values_list in args_values_lists
        )


    # if chunksize is given, and if the function object supports chunked calls,
    # then do that.
    if chunksize is not None and hasattr(fn, 'call_with_inputs'):
        if not isinstance(list_of_inputs, list):
            list_of_inputs = list(list_of_inputs)
        fn = fn.call_with_inputs
        # chunk inputs into sizes of chunksize
        list_of_inputs = [ list_of_inputs[i:i+chunksize]
                           for i in range(0, len(list_of_inputs), chunksize) ]
        total_num_inputs = len(list_of_inputs)
        #logger.debug("Chunked inputs, new list of inputs = %r", list_of_inputs)
        # and now, reset the 'chunksize' argument to multiprocessing.Pool.imap_unordered()
        chunksize = None

    if sequential_execution:
        for inp in tqdm(list_of_inputs, total=total_num_inputs):
            fn(inp)
        return

    mp_pool_imap_kwargs = {}
    if chunksize:
        mp_pool_imap_kwargs.update(chunksize=chunksize)

    mp_pool_kwargs = dict(processes=processes, initializer=_streams_init_pool)

    with multiprocessing.Pool(**mp_pool_kwargs) as pool:
        for _ in tqdm(
                pool.imap_unordered( fn, list_of_inputs, **mp_pool_imap_kwargs ),
                total=total_num_inputs
        ):
            pass
