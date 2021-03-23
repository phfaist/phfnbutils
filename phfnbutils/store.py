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

    def value_equals(self, key, test_value):
        val = self.get(key, None)
        if val is None:
            return (test_value is None)
        if isinstance(val, np.ndarray) or isinstance(test_value, np.ndarray):
            return np.all(val == test_value)
        if _normalize_attribute_value_global(val, keep_float=False) \
           != _normalize_attribute_value_global(test_value, keep_float=False):
            return False
        return True

    def __repr__(self):
        return '_Hdf5GroupProxyObject('+repr(self.grp)+')'

    def __str__(self):
        ds = {k: str(v) for k, v in self.all_attrs().items() }
        for k in self.keys_children():
            v = self.grp[k]
            ds[k] = '<{}>'.format(type(v).__name__)
        return 'HDF5 group {' + ', '.join('{}: {}'.format(k,vstr) for k,vstr in ds.items()) + '}'

    def hdf5_group(self):
        """
        Return the group object in the HDF5 data structure, giving you direct access
        to the :py:mod:`h5py` API in case you need it.
        """
        return self.grp

    def hdf5_key(self):
        """
        Return the key in the HDF5 data structure where this group is located.
        """
        return self.grp.name


def _unpack_attr_val(att_val):
    if isinstance(att_val, bytes):
        return att_val.decode('ascii')
    #if isinstance(att_val, np.ndarray) and att_val.size == 1:
    #    # if it's a scalar, return the bare scalar and not an ndarray
    #    return att_val[()]
    return att_val



def _normalize_attribute_value_global(
        value, *,
        normalize_string=_normalize_attribute_value_string,
        keep_float=True
):
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

        if self.realm not in self._store:
            # no results registered yet, nothing to yield
            return

        grp_results = self._store[self.realm]

        predicate_attrs = None
        if predicate is not None:
            sig = inspect.signature(predicate)
            predicate_attrs = list( sig.parameters.keys() )

        def want_this(grpiface):
            for k,v in kwargs.items():
                if not grpiface.value_equals(k, v):
                    return False
            if predicate is not None:
                return predicate(**{k: _unpack_attr_val(grpiface.get(k, None)) for k in predicate_attrs})
            return True

        for key in grp_results.keys():
            grp = grp_results[key]
            grpiface = _Hdf5GroupProxyObject(grp)
            if want_this(grpiface):
                yield grpiface

    def attribute_values(self, attribute_name, *, include_none=False):
        if self.realm not in self._store:
            return set()
        grp_results = self._store[self.realm]
        return set(
            _unpack_attr_val(attval)
            for attval in (
                    grp.attrs.get(attribute_name, None)
                    for grp in (grp_results[key] for key in grp_results.keys())
            )
            if include_none or attval is not None
        )
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
                     or np.issubdtype(np.dtype(type(v)), np.integer) \
                     or np.issubdtype(np.dtype(type(v)), np.floating):
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
            logger.info("Deleted results %r, key=%r", attributes, key)

    def delete_results(self, *, dry_run=False, **kwargs):
        keys_to_delete = []
        for it in self.iterate_results(**kwargs):
            keys_to_delete.append(it.hdf5_key())

        for key in keys_to_delete:
            if dry_run:
                logger.info("Delete results %r (dry run)", key)
                def _do_get_result(key):
                    # use "self" outside inner class
                    return _Hdf5GroupProxyObject(self._store[key])
                class get_all_attrs_str:
                    def __str__(self):
                        return repr(_do_get_result(key).all_attrs())
                logger.debug("with properties: %r -> %s", key, get_all_attrs_str())
            else:
                del self._store[key]
                logger.info("Deleted results %r", key)


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

    def _normalize_attribute_value(self, value, **kwargs):
        return _normalize_attribute_value_global(value, **kwargs)

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


class MultipleResults:
    def __init__(self, results=None):
        # results = [
        #   ({attrs1...}, {infoattrs1...}, <result1>),
        #   ({attrs2...}, {infoattrs2...}, <result2>),
        #   ...
        # ]
        # arttrsN are merged with "global" attributes (items
        # in attrsN take precedence)
        if results is not None:
            self.results = results #[ (attrs, info, result) for (attrs, info, result) in results ]
        else:
            self.results = []

    def append_result(self, attrs, info, result):
        # if result is itself a MultipleResults instance, merge results.
        if isinstance(result, MultipleResults):
            for res_attrs, res_info_v, res in result.results:
                try:
                    the_res_attrs = dict(attrs)
                    the_res_attrs.update(**res_attrs)
                    the_res_info = dict(info)
                    if res_info_v:
                        the_res_info.update(**res_info_v)
                    self.results.append( (the_res_attrs, the_res_info, res,) )
                except Exception as e:
                    logger.warning(
                        f"Couldn't save result {attrs}, {res_attrs}; "
                        f"[info {info}, {res_info_v}] [result {res}]: {e}"
                    )
        else:
            self.results.append( (attrs, info, result) )



class _ShowValueShort:
    def __init__(self, value, process_value=None):
        self.value = value
        self.process_value = process_value

    def _processed_value(self):
        if self.process_value is not None:
            return self.process_value(self.value)
        else:
            return self.value

    def __str__(self):
        return _showvalue(self._processed_value())

    def __repr__(self):
        return repr(self._processed_value())


def _showvalue(value, short=False):
    if isinstance(value, dict) and not short:
        return '{' + ",".join(
            "{}={}".format(k, _showvalue(v, short=True))
            for k,v in value.items()
        ) + '}'
    if short and isinstance(value, (np.ndarray,)):
        # print short version of ndarray
        with np.printoptions(precision=4,threshold=8,linewidth=9999,):
            return str(value)
    if isinstance(value, (float,)) or np.issubdtype(type(value), np.floating):
        return "%.4g"%(value)
    if value is None or isinstance(value, (int, bool, str, bytes)):
        return str(value)
    return '<{}>'.format(value.__class__.__name__)



def _call_with_accepted_kwargs(fun, kwargs):
    sig = inspect.signature(fun)
    fun_args = set( sig.parameters.keys() )
    return fun(**{k: v
                  for k, v in kwargs.items()
                  if k in fun_args})




class FnComputer:
    decode_inputargs = None
    fixed_attributes = None
    multiple_attribute_values = None
    info = None
    force_recompute = False
    skip_store = False

    def __call__(self):
        raise RuntimeError("You need to reimplement the __call__() function")


class ComputeAndStore:
    """
    Wraps a function `fn` that computes something potentially expensive with the
    necessary code to perform the computation only if it doesn't already exist
    in the data storage described by `store_filename` and `realm` and designed
    to be managed by a :py:class:`HDF5StoreResultsAccessor`.
    
    To determine whether the computation must be run, and to store the result
    after the computation if it was carried out, the attributes that
    characterize the associated result in the
    :py:class:`HDF5StoreResultsAccessor` are determined as follows (for use with
    :py:meth:`HDF5StoreResultsAccessor.has_result()` and
    :py:meth:`HDF5StoreResultsAccessor.store_result()`).  The function's named
    arguments are considered as attributes, and they are merged with the given
    attribute dictionary `fixed_attributes`.

    The return value of the function (usually a dictionary) is then stored using
    a :py:class:`HDF5StoreAccessor` instance in the given filename and realm,
    with the associated attributes.  The function may also return an instance of
    :py:class:`MultipleResults`—see more on this topic below.

    The `info` argument can be a dictionary of values to store alongside with
    the result, but that do not contribute to the identification of the result
    instance (see :py:meth:`HDF5StoreAccessor.store_result()`'s `info=` keyword
    argument).

    It is possible to "decode" some arguments of `fn()` if you would like the
    attribute value in the store file to have a different format or
    representation as the value actually passed on to `fn()`.  Use the
    `decode_inputargs()` for this purpose.  It is given the tuple of input
    arguments as-is (without any 'multiple-attributes' arguments—see below), and
    is supposed to return the arguments to send to `fn()` instead (either as a
    tuple or as a kwargs dictionary).  If a tuple is returned, it must preserve
    the order and number of the arguments.

    The results storage file `store_filename` is accessed with a
    :py:class:`HDF5StoreResultsAccessor` instance.  The instance is only created
    momentarily to check whether the results exist in the storage, and again if
    necessary to store the result into the cache.  In this way multiple
    instances of this function can run in different processes without locking
    out the results storage file.

    Messages are logged to the given `logger` instance (see python's
    :py:mod:`logging` mechanism), or to a default logger.


    **Computing functions with multiple attribute values at in one function
    call:**

    Sometimes we want to compute multiple result objects in one go, especially
    if they share some common intermediate steps.  In such cases, the function
    should return a :py:class:`MultipleResults` instance that collects the
    different result objects along with their different attributes values.  The
    attributes specified in each object in `MultipleResults` are merged with the
    function's arguments and with the `fixed_attributes`.

    When the function returns multiple result objects, then `ComputeAndStore`
    needs additional information in order to determine if a computation needs to
    run, and if so, which of those multiple results need to be computed.  Use
    the `multiple_attribute_values` field to this effect.  This field should be
    a list of dictionaries, or a dictionary containing a list in one of its
    values, that specify any additional attribute(s) and the values associated
    with the results that the function is expected to return.  These values are
    used to check the existence of the result objects in the store.

    If the function accepts a keyword argument associated with a "multiple
    result attributes", then a list of all the values that we need to compute
    (i.e., that are not in the store) is provided to the function via that
    keyword argument.  If multiple such arguments are accepted, then all these
    keyword arguments `kw1`, `kw2`, ... are given a list of the same length,
    such that `{kw1=kw1[j], kw2=kw2[j], ...}` for `j=0,1,...` describe the
    result objects that need to be computed.
    """
    def __init__(self, fn, store_filename, *,
                 realm=None,
                 fixed_attributes=None,
                 info=None,
                 decode_inputargs=None,
                 multiple_attribute_values=None,
                 force_recompute=False,
                 skip_store=False,
                 logger=None):
        self.fn = fn

        if isinstance(fn, FnComputer):
            self.fn_name = fn.__class__.__name__
            fn_sig = inspect.signature(fn.__call__)
        else:
            self.fn_name = fn.__name__
            fn_sig = inspect.signature(fn)
        self.fn_arg_names = list( fn_sig.parameters.keys() )

        self.store_filename = store_filename
        self.realm = realm

        self.fixed_attributes = {}
        if getattr(fn, 'fixed_attributes', None) is not None:
            self.fixed_attributes.update(fn.fixed_attributes)
        if fixed_attributes is not None:
            self.fixed_attributes.update(fixed_attributes)

        self.info = {}
        if getattr(fn, 'info', None) is not None:
            self.info.update(fn.info)
        if info is not None:
            self.info.update(info)

        self.decode_inputargs = None
        if getattr(fn, 'decode_inputargs', None) is not None:
            self.decode_inputargs = fn.decode_inputargs
        if decode_inputargs is not None:
            if self.decode_inputargs is not None:
                raise ValueError("decode_inputargs=... specified both in FnComputer class "
                                 "and as argument to ComputeAndStore()")
            self.decode_inputargs = decode_inputargs

        self.multiple_attribute_values = None
        if getattr(fn, 'multiple_attribute_values', None) is not None:
            self.multiple_attribute_values = fn.multiple_attribute_values
        if multiple_attribute_values is not None:
            if self.multiple_attribute_values is not None:
                raise ValueError("multiple_attribute_values=... specified both in FnComputer "
                                 "class and as argument to ComputeAndStore()")
            self.multiple_attribute_values = multiple_attribute_values
        if self.multiple_attribute_values is None:
            self.multiple_attribute_values = []
        # go through multiple_attribute_values, and replace dictionary-of-list
        # by list-of-dictionaries, i.e. {'a': [1, 2]} -> [{'a': 1}, {'a': 2}]
        self.multiple_attribute_values = \
            flatten_attribute_value_lists(self.multiple_attribute_values)
        self.multiple_attribute_all_keys = \
            list(set( itertools.chain.from_iterable(
                    d.keys() for d in self.multiple_attribute_values
                ) ))

        #print(f"{self.multiple_attribute_values=}")

        self.fn_attribute_names = [k for k in self.fn_arg_names
                                   if k not in self.multiple_attribute_all_keys ]

        self.force_recompute = False
        if hasattr(fn, 'force_recompute'):
            self.force_recompute = fn.force_recompute
        if force_recompute:
            self.force_recompute = True

        self.skip_store = False
        if hasattr(fn, 'skip_store'):
            self.skip_store = fn.skip_store
        if not skip_store:
            self.skip_store = False

        if logger is None:
            self.logger = logging.getLogger(__name__ + '.ComputeAndStore')
        else:
            self.logger = logger

    def _prepare_inputargs_as_kwargs(self, inputargs):
        decoded_inputargs = inputargs
        if self.decode_inputargs is not None:
            decoded_inputargs = self.decode_inputargs(inputargs)
        if isinstance(decoded_inputargs, dict):
            kwargs = decoded_inputargs
        else:
            if len(decoded_inputargs) != len(self.fn_attribute_names):
                raise ValueError("Can't match (decoded) input arguments %r to "
                                 "function parameters %r"
                                 % (decoded_inputargs, self.fn_attribute_names))
            kwargs = dict(zip(self.fn_attribute_names, decoded_inputargs))

        return kwargs


    def __call__(self, inputargs):
        return self.call_with_inputs( [inputargs] )

    def call_with_inputs(self, list_of_inputargs):
        logger = self.logger

        import phfnbutils # TimeThis

        if self.skip_store:
            # offer friendly warning to make sure the user didn't forget to
            # unset skip_store before a very long computation
            logger.warning("`skip_store` is set to True, results will not be stored at the end!")

        # we might have to decode the inputargs, in case they have attribute
        # values encoded in some way (e.g. dependent attributes zipped together)
        kwargs = None

        list_of_kwargs = [ self._prepare_inputargs_as_kwargs(inputargs)
                           for inputargs in list_of_inputargs ]

        list_of_kwargs_and_attributes = [
            (kwargs, dict(self.fixed_attributes, **kwargs))
            for kwargs in list_of_kwargs
        ]
        #logger.debug("requested %s(%r)", self.fn_name,
        #             _ShowValueShort(list_of_kwargs_and_attributes, lambda x: [y[1] for y in x]))


        with self._get_store() as store:

            # def is_need_to_recompute(attributes):
            #     if self.force_recompute:
            #         return True
            #     return not store.has_result(attributes)
            #
            # def which_attributes_need_recompute


            list_of_kwargs_and_attributes_and_multiattribs = []

            for kwargs, attributes in list_of_kwargs_and_attributes:
                
                multiple_attribute_values = self.multiple_attribute_values
                if not multiple_attribute_values:
                    multiple_attribute_values = [ {} ]

                # here we use multiple_attribute_values also for functions that
                # don't explicitly have any multiple_attribute_values.  In
                # thoses cases an empty list means that there is nothing to
                # compute, and a list containing only an empty dictionary means
                # that we should compute that function.

                if not self.force_recompute:
                    multiple_attribute_values = [
                        m
                        for m in multiple_attribute_values
                        if not store.has_result(dict(attributes, **m))
                    ]
                
                if not multiple_attribute_values:
                    # nothing to compute even for non-multiple-attributed
                    # functions, see comment above
                    logger.debug("Results for %s [%s] already present, not repeating computation",
                                 _ShowValueShort(attributes),
                                 _ShowValueShort(self.multiple_attribute_values))
                    continue

                multiattribkwargs = {
                    k: [m.get(k, None) for m in multiple_attribute_values]
                    for k in self.multiple_attribute_all_keys
                }

                list_of_kwargs_and_attributes_and_multiattribs.append(
                    (kwargs, attributes, multiattribkwargs)
                )

                # if not self.multiple_attribute_values:
                #     if is_need_to_recompute(attributes):
                # def have_all_necessary_results_in_store():
                #     if not self.multiple_attribute_values:
                #         return store.has_result(attributes)
                #     return 
                # if not self.force_recompute and have_all_necessary_results_in_store():
                #     logger.debug("Results for %s already present, not repeating computation",
                #                  _ShowValueShort(attributes))
                # else:
                #     new_list_of_kwargs_and_attributes.append( (kwargs,attributes,) )
            if not list_of_kwargs_and_attributes_and_multiattribs:
                logger.debug("There's nothing to compute.")
                return

        all_results = MultipleResults()

        for kwargs, attributes, multiattribkwargs \
            in list_of_kwargs_and_attributes_and_multiattribs:

            logger.info("computing for attributes = %s  [with multi-attributes = %s]",
                        _ShowValueShort(attributes), _ShowValueShort(multiattribkwargs))

            run_kwargs = dict(kwargs, **{k: v for (k,v) in multiattribkwargs.items()
                                         if k in self.fn_arg_names})

            tr = {}
            result = None
            try:
                with phfnbutils.TimeThis(tr, silent=True):
                    # call the function that actually computes the result
                    result = self.fn(**run_kwargs)
            except NoResultException as e:
                logger.warning(
                    "No result (NoResultException): %s  [for %s after %s seconds]",
                    e, _ShowValueShort(attributes), tr['timethisresult'].dt,
                )
                return False
            except Exception as e:
                logger.error("Exception while computing result!", exc_info=True)
                return False

            dt = tr['timethisresult'].dt

            if result is None:
                logger.warning("No result (returned None) for %s, after %s seconds",
                               _ShowValueShort(attributes), dt)
                return False

            logger.debug("result: %s", _ShowValueShort(result))
            logger.info("Got result for %s [runtime: %s seconds]",
                        _ShowValueShort(attributes), dt)

            the_info = {}
            for info_k, info_v in self.info.items():
                if callable(info_v):
                    info_v = _call_with_accepted_kwargs(info_v, attributes)
                the_info[info_k] = info_v
            the_info.update(timethisresult=dt)

            all_results.append_result(attributes, the_info, result)

        # store results
        if not self.skip_store:
            with self._get_store() as store:
                for attributes, the_info, result in all_results.results:
                    store.store_result(attributes, result, info=the_info)

        # signal to caller that we've computed (a) new result(s) -- but this
        # return value is probably ignored anyways
        return True

    def _get_store(self):
        store_kwargs = {}
        if self.realm is not None:
            store_kwargs.update(realm=self.realm)
        return Hdf5StoreResultsAccessor(self.store_filename, **store_kwargs)





def flatten_attribute_value_lists(alist):
    # {'a': [1, 2]} -> [{'a': 1}, {'a': 2}] for all keys in all listed dictionaries
    if isinstance(alist, dict):
        alist = [alist]
    need_another_loop = True
    while need_another_loop:
        #print(f"Looping to flatten attribute value lists, {alist=}")
        newalist = []
        need_another_loop = False
        for a in alist:
            #print(f"Inspecting {a=}")
            assert isinstance(a, dict) # should be dict here
            k, v = next( ((k, v) for (k,v) in a.items() if isinstance(v, list)),
                         (None,None) )
            if k is not None:
                #print(f"Expanding {k=}: {v=}")
                need_another_loop = True
                # expand list value into list of dictionaries with each value
                def _updated_k_with_vitem(vitem):
                    d = dict(a)
                    d[k] = vitem
                    return d
                expanded = [
                    _updated_k_with_vitem(vitem)
                    for vitem in v
                ]
                #print(f"{expanded=}") # DEBUG
                newalist += expanded
            else:
                newalist += [a] # ok, keep this dict as is
        alist = newalist
    return newalist


