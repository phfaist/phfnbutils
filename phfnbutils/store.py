import os
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

import numpy as np

import inspect
import datetime
import hashlib
import functools

import h5py
import filelock

import multiprocessing
import itertools
import random

from tqdm.auto import tqdm


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
    """
    TODO: Doc.....

    Note: must be used in a context manager!
    """
    def __init__(self, filename, *, realm='results'):
        super().__init__()

        self.filename = filename
        self.realm = realm

        self._lock_file_name = os.path.join(
            os.path.dirname(filename),
            '.' + os.path.basename(filename) + '.py_lock'
            )

        self._filelock = None
        self._store = None

        self.store_value_filters = []

    def __enter__(self):

        self._filelock = filelock.FileLock(self._lock_file_name)
        self._filelock.acquire()

        try:
            self._store = h5py.File(self.filename, 'a')
        except Exception:
            self._filelock.release()
            raise

        return self

    def __exit__(self, type, value, traceback):
        try:
            if self._store is not None:
                self._store.close()
                self._store = None
        finally:
            if self._filelock is not None:
                self._filelock.release()
                self._filelock = None

    def iterate_results(self, *, predicate=None, **kwargs):
        grp_results = self._store[self.realm]

        predicate_attrs = None
        if predicate is not None:
            sig = inspect.signature(predicate)
            predicate_attrs = list( sig.parameters.keys() )

        def want_this(grp):
            if not np.all([ self._normalize_attribute_value(grp.attrs[k], keep_float=False)
                            == self._normalize_attribute_value(v, keep_float=False)
                            for k,v in kwargs.items() ]):
                return False
            if predicate is not None:
                return predicate(**{k: _unpack_attr_val(grp.attrs[k]) for k in predicate_attrs})
            return True

        for key in grp_results.keys():
            grp = grp_results[key]
            if want_this(grp):
                yield _Hdf5GroupProxyObject(grp)

    def attribute_values(self, attribute_name):
        grp_results = self._store[self.realm]
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
                raise ValueError("key {!r} already exists in {}, not overwriting".format(key, self.realm))
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

    def delete_result(self, attributes, *, dry_run=False):
        key = self._store_key(attributes)

        if key not in self._store:
            raise ValueError("No such key for attributes {!r}".format(attributes))

        if dry_run:
            logger.info("Delete results %r, key=%r (dry run)", attributes, key)
        else:
            del self._store[key]


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

        grp_results = self._store[self.realm]

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
        return '{}/{}'.format(self.realm, the_hash)




class NoResultException(Exception):
    pass



class ComputeAndStore:
    def __init__(self, fn, store_filename, *,
                 realm=None, fixed_attributes=None, info=None,
                 force_recompute=False, logger=None):
        self.fn = fn
        self.fn_name = fn.__name__
        fn_sig = inspect.signature(fn)
        self.fn_arg_names = list( fn_sig.parameters.keys() )

        self.store_filename = store_filename
        self.realm = realm

        if fixed_attributes is None:
            self.fixed_attributes = {}
        else:
            self.fixed_attributes = fixed_attributes
        if info is None:
            self.info = {}
        else:
            self.info = info

        self.force_recompute = force_recompute

        if logger is None:
            self.logger = logging.getLogger(__name__ + '.ComputeAndStore')
        else:
            self.logger = logger


    def __call__(self, inputargs):
        fn = self.fn
        fn_name = self.fn_name
        fn_arg_names = self.fn_arg_names
        fixed_attributes = self.fixed_attributes
        force_recompute = self.force_recompute
        info = self.info
        logger = self.logger

        import phfnbutils # TimeThis

        kwargs = dict(zip(fn_arg_names, inputargs))
        attributes = dict(fixed_attributes)
        attributes.update(kwargs)
        logger.debug("requested %s(%r)", fn_name, kwargs)

        with self._get_store() as store:
            if not force_recompute and store.has_result(attributes):
                logger.debug("Results for %r already present, not repeating computation", attributes)
                return

        logger.info("computing for attributes = %r", attributes)

        tr = {}
        result = None
        try:
            with phfnbutils.TimeThis(tr):
                # call the function that actually computes the result
                result = fn(**kwargs)
        except NoResultException as e:
            logger.warning("No result could be obtained for %r, after %s seconds: %r",
                           attributes, tr['timethisresult'].dt, e)
            return False

        dt = tr['timethisresult'].dt

        if result is None:
            logger.warning("No result returned (None) for %r, after %s seconds",
                           attributes, dt)
            return

        logger.info("Result = %r [for %r, runtime %s seconds]",
                    result, attributes, dt)

        the_info = dict(info)
        the_info.update(timethisresult=dt)
        with self._get_store() as store:
            store.store_result(attributes, result, info=the_info)

        # signal to caller that we've computed a new result -- but this
        # return value is probably ignored anyways
        return True

    def _get_store(self):
        store_kwargs = {}
        if self.realm is not None:
            store_kwargs.update(realm=self.realm)
        return Hdf5StoreResultsAccessor(self.store_filename, **store_kwargs)

