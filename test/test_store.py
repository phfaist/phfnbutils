import unittest
import tempfile
import os.path

import multiprocessing

import h5py

import numpy as np

from phfnbutils.store import (
    _Hdf5GroupProxyObject, Hdf5StoreResultsAccessor, ComputeAndStore, NoResultException
)


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
        self.assertTrue(np.all(_Hdf5GroupProxyObject(handle['group1/group2']).get('data2', 13) == np.zeros((3,2))))
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
        self.assertEqual( proxy1['group2'].all_attrs(), {'attribute_1': 'Hello world', 'attribute_2': 32} )



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
        with self.assertRaises(KeyError):
            with Hdf5StoreResultsAccessor(storefn, realm='alternate_universe') as store:
                values = set([(obj['k'],obj['dt'],obj['state'],obj['variables']['Z'])
                              for obj in store.iterate_results()])

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






def global_fn(a, b, c):
    if a == 0:
        return None # test that we don't store `None`
    if b == -1:
        raise NoResultException() # couldn't get a result -- like returning None
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
                raise NoResultException() # couldn't get a result -- like returning None
            return {'res': a*10000 + b*100 + c, 'values': np.array([a,b,c])}

        compute_something = ComputeAndStore(fn, storefn,
                                            realm='somethings',
                                            fixed_attributes={'state': 'GHZ'},
                                            info={'n': 10})


        record_calls.clear()
        compute_something( (11, 22, 33) )
        self.assertEqual(record_calls, ['compute_something(11,22,33)'])

        record_calls.clear()
        compute_something( (99, 88, 77) )
        self.assertEqual(record_calls, ['compute_something(99,88,77)'])

        record_calls.clear()
        compute_something( (11, 22, 33) )
        self.assertEqual(record_calls, [])

        record_calls.clear()
        compute_something( (0, 0, 0) )
        self.assertEqual(record_calls, ['compute_something(0,0,0)'])

        record_calls.clear()
        compute_something( (0, -1, 0) )
        self.assertEqual(record_calls, ['compute_something(0,-1,0)'])

        with Hdf5StoreResultsAccessor(storefn, realm='somethings') as store:
            self.assertEqual( set([ r['res']
                                    for r in store.iterate_results() ]) ,
                              set([ 112233, 998877 ]) )



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
            ( 0, -1,  0),
        ]
            
        with multiprocessing.Pool(processes=4) as pool:
            for _ in pool.imap_unordered( compute_something, list_of_inputs ):
                pass

        with Hdf5StoreResultsAccessor(storefn, realm='yadayada') as store:
            self.assertEqual( set([ r['res']
                                    for r in store.iterate_results() ]) ,
                              set([ 112233, 445566, 778899, 10203 ]) )

