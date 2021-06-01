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

import numpy as np  # np.prod




# see https://github.com/ipython/ipython/issues/11049#issue-306086846
def _streams_init_pool():
    sys.stdout.write(".\b") # need to send *something* over stdout to use ipython's display()
    sys.stdout.flush()



def get_iter_over_all_inputs_via_arglists(args_values_lists):
    # sanity check:
    if isinstance(args_values_lists[0][0], dict):
        raise ValueError("I expected a list or iterables for input arguments, but I got a "
                         "dictionary. Did you mean to specify the `input=` keyword "
                         "argument instead?")
    iterable = itertools.chain.from_iterable(
        (itertools.product(*args_values_list) for args_values_list in args_values_lists)
    )
    total_n = sum(
        int(np.prod( [len(x) for x in args_values_list] ))
        for args_values_list in args_values_lists
    )
    return total_n, iterable

def _d_val_to_list_of_values(dval):
    if isinstance(dval, str):
        # special case for string, because it's iterable but we don't want to
        # iterate over its characters!!
        return [dval]
    try:
        return list(iter(dval))
    except TypeError:
        return [dval]

def get_iter_over_all_inputs_via_dicts(inputs):
    d_iters = [] # prepare an iterator for each dictionary in inputs
    total_n = 0
    for d in inputs:
        keys, vals = zip(*d.items())
        valiters = [_d_val_to_list_of_values(val) for val in vals]
        vals_iter = itertools.product(*valiters)
        def get_d_iter(keys=keys, vals_iter=vals_iter):
            # make sure we freeze the value of keys and vals_iters for this
            # generator
            return ( dict(zip(keys, v)) for v in vals_iter )
        d_iters.append( get_d_iter(keys, vals_iter) )
        total_n += int(np.prod( [len(x) for x in valiters] ))
        
    return total_n, itertools.chain(*d_iters)


def get_inputs_iterable(*, args_values_lists=None, inputs=None, shuffle_tasks=False):
    r"""
    Construct an iterator over all specified input combinations.

    We are given a specification of combinations of input values as expected in
    `parallel_apply_func_on_input_combinations()`.  This function returns a
    two-item tuple `(total_n, iterable)`, where `total_n` is the total number of
    input combinations, and iterable iterates over these input combinations.

    If `shuffle_tasks` is `True`, then the iterable is converted to a list and
    its order is shuffled.
    """
    if args_values_lists:
        if inputs is not None:
            raise ValueError(
                "Cannot specify both input argument lists and `inputs` keyword argument "
                "to get_inputs_iterable()."
            )
        total_n, iterable = get_iter_over_all_inputs_via_arglists(args_values_lists)
    else:
        total_n, iterable = get_iter_over_all_inputs_via_dicts(inputs)

    if shuffle_tasks:
        list_of_inputs = list(iterable)
        random.shuffle(list_of_inputs) # shuffle in place
        return total_n, list_of_inputs

    return total_n, iterable

#
# Tiny utility for applying a function using multiprocessing.Pool across a
# cartesian product of input arguments, with progress bar
#
def parallel_apply_func_on_input_combinations(
        fn,
        *args_values_lists,
        #
        inputs=None,
        processes=None,
        shuffle_tasks=True,
        sequential_execution=False,
        chunksize=None,
):
    # Notes: (convert to docstring, TODO)
    #
    # - fn should accept a single argument, a tuple of arguments (if
    #   `*args_values_lists` is specified) or a dictionary (if `inputs=`) is
    #   specified.
    #
    # - if `inputs` is non-`None`, it should be a list of dictionaries where
    #   each value itself can be a single value or a list.  For one of these
    #   dictionaries, an associated list of input values is assembled by taking
    #   all combinations of the values within the dictionary.  The resulting
    #   lists are then concatenated to obtain the list of all inputs.
    #
    # - shuffle_tasks shuffles the order of execution of all tasks; time
    #   progress reporting might be more reliable than if all the hard tasks are
    #   at the end of the values lists.  A cost is that the iterable
    #   representing the cartesian product of input arguments must be flattened
    #   to a list, possibly using lots of memory if you have many input
    #   arguments with many values.
    
    from tqdm.auto import tqdm

    if len(args_values_lists) > 0 and inputs is not None:
        raise ValueError("Cannot specify both input argument lists "
                         "and `inputs` keyword argument to "
                         "parallel_apply_func_on_input_combinations().")

    total_num_inputs, iterable_of_inputs = get_inputs_iterable(
        args_values_lists=args_values_lists,
        inputs=inputs,
        shuffle_tasks=shuffle_tasks,
    )

    # if chunksize is given, and if the function object supports chunked calls,
    # then do that.
    if chunksize is not None and hasattr(fn, 'call_with_inputs'):
        if not isinstance(iterable_of_inputs, list):
            iterable_of_inputs = list(iterable_of_inputs)
        #
        # Instead of calling fn(input), we will call
        # fn.call_with_inputs([input1, input2, ...]) with all the inputs in a
        # given chunk.  So what we do is we simply "redefine" what the function
        # is and what the inputs are.
        #
        #  fn                 ->     fn.call_with_inputs
        #  list_of_inputs     ->     [  [in_chunk1_1, in_chunk1_2, ... ],
        #                               [in_chunk2_1, in_chunk2_2, ... ], ... ]
        #
        #

        fn = fn.call_with_inputs
        # chunk inputs into sizes of chunksize
        iterable_of_inputs = [ iterable_of_inputs[i:i+chunksize]
                               for i in range(0, len(iterable_of_inputs), chunksize) ]
        total_num_inputs = len(iterable_of_inputs)
        #logger.debug("Chunked inputs, new list of inputs = %r", iterable_of_inputs)
        # and now, reset the 'chunksize' argument to multiprocessing.Pool.imap_unordered()
        chunksize = None

    if sequential_execution:
        for inp in tqdm(iterable_of_inputs, total=total_num_inputs):
            fn(inp)
        return

    mp_pool_imap_kwargs = {}
    if chunksize:
        mp_pool_imap_kwargs.update(chunksize=chunksize)

    mp_pool_kwargs = dict(processes=processes, initializer=_streams_init_pool)

    with multiprocessing.Pool(**mp_pool_kwargs) as pool:
        for _ in tqdm(
                pool.imap_unordered( fn, iterable_of_inputs, **mp_pool_imap_kwargs ),
                total=total_num_inputs
        ):
            pass
