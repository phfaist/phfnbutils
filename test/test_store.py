import unittest
import tempfile
import os.path

import itertools

import multiprocessing

import h5py

import logging

import numpy as np

from phfnbutils.store import (
    _Hdf5GroupProxyObject, Hdf5StoreResultsAccessor, ComputeAndStore, FnComputer,
    MultipleResults, NoResultException,
    flatten_attribute_value_lists
)


# ------------------------------------------------------------------------------


class TestProxyObject(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name

        handle = h5py.File(os.path.join(self.temp_dir_name, 'temp.hdf5'), 'a')
        
        handle.create_group('group1')
        d1 = handle['group1'].create_dataset('data', (1,10,))
        d1[:,:] = np.arange(10)
        handle.create_group('group1/group2')
        d2 = handle['group1/group2'].create_dataset('data2', (3,2,))
        d2[:,:] = np.zeros((3,2))
        handle['group1/group2'].attrs['attribute_1'] = "Hello world".encode('ascii')
        handle['group1/group2'].attrs['attribute_2'] = 32

        self.handle = handle

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_proxy_getitem(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])
        self.assertIsInstance(proxy1['group2'], _Hdf5GroupProxyObject)
        self.assertTrue(np.all(proxy1['data'] == np.arange(10)))
        self.assertTrue(np.all(proxy1['group2']['data2'] == np.zeros((3,2))))
        self.assertTrue(np.all(_Hdf5GroupProxyObject(handle['group1/group2'])['data2'] == np.zeros((3,2))))
        self.assertEqual(proxy1['group2']['attribute_2'], 32)
        self.assertEqual(proxy1['group2']['attribute_1'], 'Hello world') # as string
        with self.assertRaises(KeyError):
            value = proxy1['key-does-not-exist']

    def test_proxy_get(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])
        self.assertIsInstance(proxy1.get('group2', None), _Hdf5GroupProxyObject)
        self.assertTrue(np.all(proxy1.get('data', None) == np.arange(10)))
        self.assertIsNone(proxy1.get('data_does_not_exist', None))
        self.assertTrue(np.all(proxy1.get('group2', {}).get('data2', None) == np.zeros((3,2))))
        self.assertTrue(np.all(_Hdf5GroupProxyObject(handle['group1/group2']).get('data2', 13)
                               == np.zeros((3,2))))
        self.assertEqual(proxy1['group2'].get('attribute_2', 1000), 32)
        self.assertEqual(proxy1['group2'].get('attribute_1', 'hi there'), 'Hello world') # as string
        self.assertEqual(proxy1['group2'].get('attribute_zzzz', 'hi there'), 'hi there')
        self.assertEqual(proxy1.get('key-does-not-exist', 123), 123)

    def test_proxy_keys(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])

        self.assertEqual( set(proxy1.keys()), set(['group2', 'data']) )
        self.assertEqual( set(proxy1['group2'].keys()), set(['attribute_1', 'attribute_2', 'data2']) )

        self.assertEqual( set(proxy1.keys_children()), set(['group2', 'data']) )
        self.assertEqual( set(proxy1['group2'].keys_children()), set(['data2']) )

        self.assertEqual( set(proxy1.keys_attrs()), set([]) )
        self.assertEqual( set(proxy1['group2'].keys_attrs()), set(['attribute_1', 'attribute_2']) )

    def test_proxy_all_attrs(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])

        self.assertEqual( proxy1.all_attrs(), {} )
        self.assertEqual( proxy1['group2'].all_attrs(),
                          {'attribute_1': 'Hello world', 'attribute_2': 32} )

    def test_proxy_str(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])
        self.assertEqual(str(proxy1), "HDF5 group {data: <Dataset>, group2: <Group>}")
        self.assertEqual(str(proxy1['group2']),
                         "HDF5 group {attribute_1: Hello world, attribute_2: 32, data2: <Dataset>}")

    def test_proxy_value_equals(self):
        handle = self.handle
        proxy1 = _Hdf5GroupProxyObject(handle['group1'])
        self.assertTrue(proxy1.value_equals('data', [0,1,2,3,4,5,6,7,8,9]))
        self.assertTrue(proxy1['group2'].value_equals('attribute_2', 32))
        self.assertTrue(proxy1['group2'].value_equals('attribute_1', 'Hello world'))
        self.assertTrue(proxy1['group2'].value_equals('data2', np.zeros((3,2,))))




# ------------------------------------------------------------------------------



class TestStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_store_and_retrieve(self):

        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(3,1,'GHZ',2.0), (4,2,'GHZ',0.5)]))

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(k=3)]),
                             set([(3,1,'GHZ',2.0)]))

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(state='GHZ')]),
                             set([(3,1,'GHZ',2.0), (4,2,'GHZ',0.5)]))

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(n=12)]),
                             set([(3,1,'GHZ',2.0), (4,2,'GHZ',0.5)]))

        def k_is_even(k, state):
            self.assertEqual(state, 'GHZ')
            return True if k % 2 == 0 else False

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(predicate=k_is_even)]),
                             set([(4,2,'GHZ',0.5)]))


        # nothing stored in this realm yet
        with Hdf5StoreResultsAccessor(storefn, realm='alternate_universe') as store:
            self.assertEqual(
                set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                     for obj in store.iterate_results()]),
                set()
            )

        # store something in a different realm
        with Hdf5StoreResultsAccessor(storefn, realm='alternate_universe') as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'k': 4, 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'k': 5, 'method': 'exact'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 1.0} },
                                info={'dt': 1} )

        # check that we pick up results stored in this realm
        with Hdf5StoreResultsAccessor(storefn, realm='alternate_universe') as store:
            self.assertEqual(set([(obj['k'],obj['method'],obj['dt'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(4,'direct',2,0.5),(5,'exact',1,1.0)]))

        # results in other (default) realm are unaffected
        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(3,1,'GHZ',2.0), (4,2,'GHZ',0.5)]))


    def test_iterate_results_when_empty(self):

        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(
                [(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                 for obj in store.iterate_results()],
                []
            )

    def test_iterate_results_inconsistent_keys(self):

        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(method='direct')]),
                             set([(4,2,'GHZ',0.5)]))


        def predicate_method_direct(method):
            return method == 'direct'

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results(predicate=predicate_method_direct)]),
                             set([(4,2,'GHZ',0.5)]))

    def test_iterate_results_by_values(self):

        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'n':  12, 'k': 5, 'state': 'GHZ', 'method': 'indirect'},
                                { 'result1': np.ones((3,4)),
                                  'kkk': np.array([5,], dtype=np.int32),
                                  'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 4} )

        # can query with `None` to test for absence of key 
        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'],)
                                  for obj in store.iterate_results(method=None)]),
                             set([(3,1,'GHZ',2.0,)]))

        # can query with predicate w/ array-vs-scalar matching
        def predicate_kkk(kkk):
            return (kkk == 5)

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'],)
                                  for obj in store.iterate_results(predicate=predicate_kkk)]),
                             set([(5,4,'GHZ',.5,)]))

    def test_attribute_values(self):

        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        # works when empty
        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual( store.attribute_values('method') ,
                              set() )

        # store some results
        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'n':  12, 'k': 5, 'state': 'GHZ', 'method': 'indirect'},
                                { 'result1': np.ones((3,4)),
                                  'kkk': np.array([5,], dtype=np.int32),
                                  'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 4} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(
                store.attribute_values('k'),
                set([3, 4, 5])
            )

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(
                store.attribute_values('method'),
                set(['direct', 'indirect'])
            )

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(
                store.attribute_values('method', include_none=True),
                set(['direct', 'indirect', None])
            )


    def test_delete_result(self):
        
        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            store.delete_result({'n': 12, 'k': 4, 'state': 'GHZ',
                                 'method': 'direct'})

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(3,1,'GHZ',2.0)]))


    def test_delete_results_dry_run(self):
        
        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'n':  10, 'k': 6, 'state': 'Bell'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 3} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            store.delete_results(n=12, dry_run=True)

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(6,3,'Bell',2.0),
                                  (4,2,'GHZ',0.5),
                                  (3,1,'GHZ',2.0),]))

    def test_delete_results(self):
        
        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'n':  10, 'k': 6, 'state': 'Bell'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 3} )

        with Hdf5StoreResultsAccessor(storefn) as store:
            store.delete_results(n=12)

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(6,3,'Bell',2.0)]))

    def test_delete_results_with_predicate(self):
        
        storefn = os.path.join(self.temp_dir_name, 'temptest.hdf5')

        with Hdf5StoreResultsAccessor(storefn) as store:
            # store.store_result(attributes, result, ...)
            store.store_result( {'n':  12, 'k': 3, 'state': 'GHZ'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 1} )
            store.store_result( {'n':  12, 'k': 4, 'state': 'GHZ', 'method': 'direct'},
                                { 'result1': np.ones((3,4)), 'variables': {'R': 4*np.arange(12), 'Z': 0.5} },
                                info={'dt': 2} )
            store.store_result( {'n':  10, 'k': 6, 'state': 'Bell'},
                                { 'result1': np.zeros((3,4)), 'variables': {'R': np.arange(12), 'Z': 2.0} },
                                info={'dt': 3} )

        def predicate(n):
            return (n == 12)

        with Hdf5StoreResultsAccessor(storefn) as store:
            store.delete_results(predicate=predicate)

        with Hdf5StoreResultsAccessor(storefn) as store:
            self.assertEqual(set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                                  for obj in store.iterate_results()]),
                             set([(6,3,'Bell',2.0)]))






# ------------------------------------------------------------------------------


def global_fn(a, b, c):
    if a == 0:
        return None # test that we don't store `None`
    if b == -1:
        # couldn't get a result -- like returning None, but with a specific
        # message
        raise NoResultException("No convergence")
    if c is None:
        # test that we can handle general exceptions in the function
        raise ValueError("Invalid input: c is None")
    return {'res': a*10000 + b*100 + c, 'values': np.array([a,b,c])}



class TestComputeAndStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_and_store(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        record_calls = []

        def fn(a, b, c):
            record_calls.append("compute_something({},{},{})".format(a,b,c))
            if a == 0:
                return None # test that we don't store `None`
            if b == -1:
                # couldn't get a result -- like returning None
                raise NoResultException("this is why")
            if c is None:
                # test that we can handle general exceptions in the function
                raise ValueError("Invalid input: c is None")
            return {'res': a*10000 + b*100 + c, 'values': np.array([a,b,c])}

        compute_something = ComputeAndStore(fn, storefn,
                                            realm='somethings',
                                            fixed_attributes={'state': 'GHZ'},
                                            info={'n': 10,
                                                  'two_times_a': lambda a: 2*a })


        record_calls.clear()
        compute_something( (11, 22, 33) )
        self.assertEqual(record_calls, ['compute_something(11,22,33)'])

        record_calls.clear()
        compute_something( (99, 88, 77) )
        self.assertEqual(record_calls, ['compute_something(99,88,77)'])

        record_calls.clear()
        compute_something( (11, 22, 33) )
        self.assertEqual(record_calls, [])

        with self.assertLogs('phfnbutils.store', level=logging.WARNING):
            record_calls.clear()
            compute_something( (0, 0, 0) )
            self.assertEqual(record_calls, ['compute_something(0,0,0)'])

        with self.assertLogs('phfnbutils.store', level=logging.WARNING):
            record_calls.clear()
            compute_something( (1, -1, 0) )
            self.assertEqual(record_calls, ['compute_something(1,-1,0)'])

        with self.assertLogs('phfnbutils.store', level=logging.ERROR):
            record_calls.clear()
            compute_something( (1, 1, None) )
            self.assertEqual(record_calls, ['compute_something(1,1,None)'])

        with Hdf5StoreResultsAccessor(storefn, realm='somethings') as store:
            self.assertEqual( set([ (r['res'], r['two_times_a'])
                                    for r in store.iterate_results() ]) ,
                              set([ (112233, 22), (998877, 198) ]) )



    def test_in_multiprocessing(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        compute_something = ComputeAndStore(global_fn, storefn,
                                            realm='yadayada',
                                            fixed_attributes={'state': 'GHZ'},
                                            info={'n': 10})

        list_of_inputs = [
            (11, 22, 33),
            (44, 55, 66),
            (77, 88, 99),
            ( 1,  2,  3),
            ( 0,  0,  0),
            ( 1, -1,  0),
            ( 1,  1,  None), # ValueError
        ]
            
        # Note that the ValueError caused by input (1,1,None) is not re-raised,
        # it is only reported in the logs as an error.  This is to prevent the
        # exception from interrupting all the other computations in a lengthy
        # job.  Check for that:
        #
        #with self.assertLogs('phfnbutils.store', level=logging.ERROR):
        #
        # ### <-- this doesn't seem to work with log messages emitted inside a
        # ### mulitprocessing worker; ignore this check for now, we already test
        # ### that in the non-multiprocessing test

        with multiprocessing.Pool(processes=4) as pool:
            for _ in pool.imap_unordered( compute_something, list_of_inputs ):
                pass

        with Hdf5StoreResultsAccessor(storefn, realm='yadayada') as store:
            self.assertEqual( set([ r['res']
                                    for r in store.iterate_results() ]) ,
                              set([ 112233, 445566, 778899, 10203 ]) )


    def test_with_decode_inputargs(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        def fn(a, b, c):
            if a == 0:
                return None # test that we don't store `None`
            if b == -1:
                # couldn't get a result -- like returning None
                raise NoResultException("this is why")
            if c is None:
                # test that we can handle general exceptions in the function
                raise ValueError("Invalid input: c is None")
            return {'res': a*10000 + b*100 + c, 'values': np.array([a,b,c])}

        def decode_inputargs(inputargs, use_dict=False):
            ((b,c),a) = inputargs
            if use_dict:
                return {'a': a, 'b': b, 'c': c}
            return (a, b, c)

        compute_something = ComputeAndStore(fn, storefn,
                                            realm='foobar',
                                            fixed_attributes={'state': 'bliss'},
                                            decode_inputargs=lambda a: decode_inputargs(a),
                                            info={'n': 10})

        list_of_inputs = [
            ( (22, 33), 11 ),
            ( (55, 66), 44),
            ( (88, 99), 77),
        ]
            
        for input in list_of_inputs:
            compute_something(input)

        with Hdf5StoreResultsAccessor(storefn, realm='foobar') as store:
            self.assertEqual( set([ r['res']
                                    for r in store.iterate_results() ]) ,
                              set([ 112233, 445566, 778899 ]) )


        compute_something2 = ComputeAndStore(fn, storefn,
                                            realm='foobar2',
                                            fixed_attributes={'state': 'bliss'},
                                            decode_inputargs=\
                                             lambda a: decode_inputargs(a, use_dict=True),
                                            info={'n': 10})

        for input in list_of_inputs:
            compute_something2(input)

        with Hdf5StoreResultsAccessor(storefn, realm='foobar2') as store:
            self.assertEqual( set([ r['res']
                                    for r in store.iterate_results() ]) ,
                              set([ 112233, 445566, 778899 ]) )



    def test_compute_and_store_multiinputs(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        def fn(factor, a):
            return {'res_value': factor*a}

        compute_something = ComputeAndStore(fn, storefn,
                                            realm='somethings',
                                            fixed_attributes={'state': 'GHZ'},
                                            info={'n': 10,
                                                  'ten_times_a': lambda a: 10*a })

        compute_something.call_with_inputs( itertools.product([1,2,3,4,5], [7,8,]) )

        with Hdf5StoreResultsAccessor(storefn, realm='somethings') as store:
            #for r in store.iterate_results():
            #    print(r)
            self.assertEqual( set([ (r['a'], r['res_value'], r['factor'], r['ten_times_a'])
                                    for r in store.iterate_results() ]) ,
                              set([ (7,  7, 1, 70),
                                    (7, 14, 2, 70),
                                    (7, 21, 3, 70),
                                    (7, 28, 4, 70),
                                    (7, 35, 5, 70),
                                    (8,  8, 1, 80),
                                    (8, 16, 2, 80),
                                    (8, 24, 3, 80),
                                    (8, 32, 4, 80),
                                    (8, 40, 5, 80), ]) )



    def test_compute_and_store_multiple_results(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        def fn(a):
            mr = MultipleResults([
                ({'factor': 1}, None, {'res_value': a},),
                ({'factor': 2}, None, {'res_value': 2*a},),
                ({'factor': 3}, {'factor_is_four': False}, {'res_value': 3*a},),
            ])
            mr.append_result( {'factor': 4}, {'factor_is_four': True}, {'res_value': 4*a} )
            mr.append_result( {}, {'factor_is_four': False},
                              MultipleResults([ ({'factor': 5}, None, {'res_value': 5*a}), ]) )
            return mr

        compute_something = ComputeAndStore(fn, storefn,
                                            realm='somethings',
                                            fixed_attributes={'state': 'GHZ'},
                                            info={'n': 10,
                                                  'ten_times_a': lambda a: 10*a })

        compute_something( (7,) )
        compute_something( (8,) )

        with Hdf5StoreResultsAccessor(storefn, realm='somethings') as store:
            #for r in store.iterate_results():
            #    print(r)
            self.assertEqual( set([ (r['a'], r['res_value'], r['factor'], r['ten_times_a'])
                                    for r in store.iterate_results() ]) ,
                              set([ (7,  7, 1, 70),
                                    (7, 14, 2, 70),
                                    (7, 21, 3, 70),
                                    (7, 28, 4, 70),
                                    (7, 35, 5, 70),
                                    (8,  8, 1, 80),
                                    (8, 16, 2, 80),
                                    (8, 24, 3, 80),
                                    (8, 32, 4, 80),
                                    (8, 40, 5, 80), ]) )


    # test multi-attrib functions
    def test_multi_0(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        record_calls = []

        def fn(a, b):
            record_calls.append( (a, b) )
            return MultipleResults([
                ({'operation': '+'}, None, {'operation_result': a + b}),
                ({'operation': '*'}, None, {'operation_result': a * b}),
            ])

        compute_something = ComputeAndStore(
            fn, storefn,
            fixed_attributes={'fixattr1': 'one'},
            multiple_attribute_values={'operation': ['+','*']},
            info={'info1': 10,}
        )

        record_calls.clear()
        compute_something( (3,8) )
        self.assertEqual(record_calls, [(3,8)])

        record_calls.clear()
        compute_something( (1,2) )
        self.assertEqual(record_calls, [(1,2)])

        with Hdf5StoreResultsAccessor(storefn) as store:
            #for r in store.iterate_results():
            #    print(r)
            self.assertEqual( set([ (r['a'], r['b'], r['operation'], r['operation_result'])
                                    for r in store.iterate_results() ]) ,
                              set([ (3, 8, '+', 11),
                                    (3, 8, '*', 24),
                                    (1, 2, '+', 3),
                                    (1, 2, '*', 2),
                              ]) )

    def test_multi_1(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        record_calls = []

        def fn(a, b, *, operation):
            record_calls.append( (a, b, operation) )
            return MultipleResults([
                ({'operation': '+'}, None, {'operation_result': a + b}),
                ({'operation': '*'}, None, {'operation_result': a * b}),
            ])

        compute_something = ComputeAndStore(
            fn, storefn,
            fixed_attributes={'fixattr1': 'one'},
            multiple_attribute_values={'operation': ['+','*']},
            info={'info1': 10,}
        )

        record_calls.clear()
        compute_something( (3,4) )
        self.assertEqual(record_calls, [(3,4,['+','*'])])

        with Hdf5StoreResultsAccessor(storefn) as store:
            store.store_result(dict(fixattr1='one', a=1, b=2, operation='*'),
                               {'operation_result': 2})

        # If '*' is already in store, it should only request '+'
        record_calls.clear()
        compute_something( (1,2) )
        self.assertEqual(record_calls, [(1,2,['+'])])

        record_calls.clear()
        compute_something( (1,2) )
        self.assertEqual(record_calls, [])

    def test_multi_2(self):

        storefn = os.path.join(self.temp_dir_name, 'tempstore.hdf5')

        record_calls = []

        class fnclass(FnComputer):
            def __call__(self, a, b, operation):
                record_calls.append( (a, b, operation) )
                return MultipleResults([
                    ({'operation': '+'}, None, {'operation_result': a + b}),
                    ({'operation': '*'}, None, {'operation_result': a * b}),
                ])


        compute_something = ComputeAndStore(
            fnclass(), storefn,
            fixed_attributes={'fixattr1': 'one'},
            multiple_attribute_values={'operation': ['+','*']},
            info={'info1': 10,}
        )

        record_calls.clear()
        compute_something( (3,8) )
        self.assertEqual(record_calls, [(3,8,['+','*'])])

        with Hdf5StoreResultsAccessor(storefn) as store:
            #for r in store.iterate_results():
            #    print(r)
            self.assertEqual( set([ (r['a'], r['b'], r['operation'], r['operation_result'])
                                    for r in store.iterate_results() ]) ,
                              set([ (3, 8, '+', 11),
                                    (3, 8, '*', 24),
                              ]) )





    def test_flatten_attribute_value_lists(self):

        self.assertEqual(
            flatten_attribute_value_lists([]),
            []
        )
        self.assertEqual(
            flatten_attribute_value_lists({'a': 1, 'b': 2}),
            [{'a': 1, 'b': 2}]
        )
        self.assertEqual(
            flatten_attribute_value_lists([{'a': 1, 'b': 2}]),
            [{'a': 1, 'b': 2}]
        )
        self.assertEqual(
            flatten_attribute_value_lists({'a': 1, 'b': [3, 4]}),
            [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}]
        )
        self.assertEqual(
            flatten_attribute_value_lists({'a': [1,2], 'b': [3, 4]}),
            [{'a': 1, 'b': 3}, {'a': 1, 'b': 4},
             {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
        )





if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
