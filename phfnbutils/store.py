import os
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

import numpy as np

import inspect
import datetime
import hashlib

import h5py
import filelock

import multiprocessing
import itertools
import random

from tqdm.auto import tqdm


#
# Tiny utility for applying a function using multiprocessing.Pool across a
# cartesian product of input arguments, with progress bar
#
def parallel_apply_func_on_input_combinations(fn, args_values_list, *, processes=None, shuffle_tasks=True):
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
    
    list_of_inputs = itertools.product(*args_values_list)
    if shuffle_tasks:
        list_of_inputs = list(list_of_inputs)
        random.shuffle(list_of_inputs) # shuffle in place
    with multiprocessing.Pool(processes=processes) as pool:
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


#
# utilities for my hdf5 datasets
#
def _normalize_attribute_value_string(v):
    # NOTE: Only ASCII strings allowed in string values.
    return v.encode('ascii')

class _Hdf5GroupProxyObject:
    def __init__(self, grp):
        self.grp = grp

    def get(self, key, default, *, _default_action=None):
        if key in self.grp:
            obj = self.grp[key]
            if isinstance(obj, h5py.Group):
                return _Hdf5GroupProxyObject(self.grp[key])
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            raise ValueError("Can't interface object value {!r}".format(obj))
        if key in self.grp.attrs:
            return self._unpack_attr_val(self.grp.attrs[key])
        if _default_action:
            return _default_action()
        return default

    def keys(self):
        return itertools.chain(self.grp.keys(), self.grp.attrs.keys())

    def keys_children(self):
        return self.grp.keys()

    def keys_attrs(self):
        return self.grp.attrs.keys()

    def all_attrs(self):
        return dict([(k, self._unpack_attr_val(v)) for (k,v) in self.grp.attrs.items()])

    def __getitem__(self, key):
        def keyerror():
            raise KeyError("No key {} in hdf5 group {!r} or its attributes".format(key, self.grp))
        return self.get(key, None, _default_action=keyerror)

    def _unpack_attr_val(self, att_val):
        return _unpack_attr_val(att_val) # call global method

    def __repr__(self):
        return '_Hdf5GroupProxyObject('+repr(self.grp)+')'


def _unpack_attr_val(att_val):
    if isinstance(att_val, bytes):
        return att_val.decode('ascii')
    #if isinstance(att_val, np.ndarray) and att_val.size == 1:
    #    # if it's a scalar, return the bare scalar and not an ndarray
    #    return att_val[()]
    return att_val


class Hdf5StoreResultsAccessor:
    def __init__(self, filename):
        self._lock_file_name = os.path.join(
            os.path.dirname(filename),
            '.' + os.path.basename(filename) + '.py_lock'
            )

        self._filelock = filelock.FileLock(self._lock_file_name)
        self._filelock.acquire()

        try:
            self._store = h5py.File(filename, 'a')
        except Exception:
            self._filelock.release()
            raise

        self.store_value_filters = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self._store.close()
            self._store = None
        finally:
            self._filelock.release()
            self._filelock = None

    def iterate_results(self, *, predicate=None, **kwargs):
        grp_results = self._store['results']

        predicate_attrs = None
        if predicate is not None:
            sig = inspect.signature(predicate)
            predicate_attrs = [ param.name for param in sig.parameters ]

        def want_this(grp):
            if not np.all([ self._normalize_attribute_value(grp.attrs[k], keep_float=False)
                            == self._normalize_attribute_value(v, keep_float=False)
                            for k,v in kwargs.items() ]):
                return False
            if predicate is not None:
                return predicate(**{k: grp.attrs[k] for k in predicate_attrs})
            return True

        for key in grp_results.keys():
            grp = grp_results[key]
            if want_this(grp):
                yield _Hdf5GroupProxyObject(grp)

    def attribute_values(self, attribute_name):
        grp_results = self._store['results']
        return set( _unpack_attr_val(grp.attrs[attribute_name])
                    for grp in (grp_results[key] for key in grp_results.keys()) )
        # vals = set()
        # for key in grp_results.keys():
        #     grp = grp_results[key]
        #     this_val = _unpack_attr_val(grp.attrs[attribute_name])
        #     if this_val not in vals:
        #         vals.append(this_val)
        # return vals

    def has_result(self, attributes):
        key = self._store_key(attributes)
        if key in self._store:
            return True
        return False

    def get_result(self, attributes):
        key = self._store_key(attributes)
        if key in self._store:
            grp = self._store[key]
            return _Hdf5GroupProxyObject(grp)
        return None

    def store_result(self, attributes, value, *, forbid_overwrite=False, info=None):
        key = self._store_key(attributes)

        if key in self._store:
            if forbid_overwrite:
                raise ValueError("key {!r} already exists in results, not overwriting")
            del self._store[key]

        grp = self._store.create_group(key)

        for k, v in attributes.items():
            grp.attrs[k] = self._normalize_attribute_value(v)

        for filt in self.store_value_filters:
            value = filt(value)
        has_error = self._store_result_dict_value(grp, value)
        # only raise errors *after* having written everything to disk, in case
        # that computation was very time-costly to obtain and our poor user
        # would otherwise lose all their hard-obtained results 
        if has_error is not None:
            raise has_error

        if info:
            for k, v in info.items():
                grp.attrs[k] = self._normalize_attribute_value(v)

    def _store_result_dict_value(self, grp, value):

        has_error = None
        for k, v in value.items():
            if k.startswith('_'):
                continue
            try:
                for filt in self.store_value_filters:
                    v = filt(v)
                if v is None:
                    continue
                if isinstance(v, dict):
                    newgrp = grp.create_group(k)
                    has_error = self._store_result_dict_value(newgrp, v)
                elif isinstance(v, (np.ndarray, int, float)) \
                     or np.issubdtype(type(v), np.integer) \
                     or np.issubdtype(type(v), np.floating):
                    dset = grp.create_dataset(k, data=v)
                elif isinstance(v, (datetime.date, datetime.time, datetime.datetime)):
                    grp.attrs[k] = v.isoformat().encode('ascii')
                elif isinstance(v, (datetime.timedelta,)):
                    grp.attrs[k] = ("timedelta(seconds={:.06g})".format(v.total_seconds())).encode('ascii')
                else:
                    has_error = ValueError("Can't save object {!r}, unknown type".format(v))
                    # continue saving other stuff
            except Exception as e:
                has_error = e

        return has_error

    def update_keys(self, attribute_names, *, add_default_keys=None, dry_run=False):
        """
        Checks that all result storage keys are up-to-date.  If you introduce a new
        kwarg attribute in the storage, we can set that attribute to all
        existing results with the given value in `add_default_keys`.

        - `attribute_names` is a list or tuple of attribute names to consider when
          composing the storage key.

        - `add_default_keys` is a dictionary of new attribute names and values
          to set to records that don't have that attribute set
        """

        rename_keys = [] # [ (oldkey,newkey), ... ]
        set_attributes = {} # { newkey: {attribute1: value1 ...}, ... }

        if add_default_keys is None:
            add_default_keys = {}

        grp_results = self._store['results']

        for key in grp_results.keys():
            grp = grp_results[key]

            these_attributes = {}
            this_set_attributes = {}

            for k in attribute_names:
                att_value = None
                if k in grp.attrs:
                    att_value = grp.attrs[k]
                else:
                    if k in add_default_keys:
                        att_value = add_default_keys[k]
                        this_set_attributes[k] = att_value
                    else:
                        att_value = None
                    
                these_attributes[k] = att_value

            # also take note of any default attributes to set that are not part
            # of the results-identifying attributes
            for k, v in ((akey, aval,)
                         for akey, aval in add_default_keys.items()
                         if akey not in attribute_names):
                if k not in grp.attrs:
                    this_set_attributes[k] = v

            newkey = self._store_key(these_attributes, hash_only=True)
            if newkey != key:
                logger.debug("Will rename {} -> {}".format(key, newkey))
                rename_keys.append( (key, newkey) )

            if this_set_attributes:
                logger.debug("Will set attributes on newkey {}: {!r}".format(newkey, this_set_attributes))
                set_attributes[newkey] = this_set_attributes

        if not rename_keys and not set_attributes:
            logger.debug("All keys and attributes are up-to-date.")
            return
        
        logger.debug("Finished inspecting keys, proceeding to updates ... ")

        for oldkey, newkey in rename_keys:
            if dry_run:
                logger.info("\tgrp_results.move({!r}, {!r})".format(oldkey, newkey))
            else:
                grp_results.move(oldkey, newkey)

        for newkey, attrib in set_attributes.items():
            grp = grp_results[newkey] if not dry_run else None
            for ak, av in attrib.items():
                if dry_run:
                    logger.info("\tresults({!r}).attrs[{!r}] = {!r}".format(newkey, ak, av))
                else:
                    grp.attrs[ak] = self._normalize_attribute_value(av)

        logger.debug("Keys and attributes renamed successfully.")

    def _normalize_attribute_value(self, value, *, normalize_string=_normalize_attribute_value_string, keep_float=True):
        t = type(value)
        if value is None:
            return ""
        if isinstance(value, str):
            return _normalize_attribute_value_string(value)
        if isinstance(value, bytes):
            # bytes and str are treated the same, as ASCII strings.  For storage
            # of raw binary data you'll want to store a dataset of some kind
            # e.g. with numpy.
            return value
        if isinstance(value, int) or np.issubdtype(t, np.integer):
            return int(value)
        if isinstance(value, float) or np.issubdtype(t, np.floating):
            if keep_float:
                return value
            else:
                return _normalize_attribute_value_string( '{:0.8g}'.format(value) )
        if isinstance(value, (datetime.date, datetime.time, datetime.datetime)):
            return _normalize_attribute_value_string(value.isoformat())
        if isinstance(value, (datetime.timedelta,)):
            return _normalize_attribute_value_string("total_seconds={:.06g}".format(value.total_seconds()))

        raise ValueError("Cannot encode {!r} for HDF5 attribute storage, unknown type".format(value))

    def _store_key(self, attributes, *, hash_only=False):
        m = hashlib.sha1()
        stuff = "\n".join(
            "{key}={value}\n".format(
                key=k,
                value=repr(self._normalize_attribute_value(attributes[k], keep_float=False))
            )
            for k in sorted(attributes.keys())
        )

        m.update( stuff.encode('ascii') )
        the_hash = m.hexdigest()
        if hash_only:
            return the_hash
        return 'results/{}'.format(the_hash)


