import unittest
import tempfile
import os.path

import multiprocessing

import h5py

import logging

import numpy as np

import multiprocessing



from phfnbutils.mp import (
    parallel_apply_func_on_input_combinations,
    get_inputs_iterable
)

# ------------------------------------------------------------------------------



class CallsRecorder:
    def __init__(self, initial_list=None):
        if initial_list is not None:
            self.called_args = initial_list
        else:
            self.called_args = []

    def __call__(self, arg):
        self.called_args.append(arg)



def fn_helper(factor, a):
    return {'res_value': factor*a}




class Test_parallel_apply_func_on_input_combinations(unittest.TestCase):

    def test_0(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                [['A','B','C'], ['D','E','F']],
                [['Z'], ['Z']],
                [[0,1], [10,20]],
                shuffle_tasks=True,
                sequential_execution=True,
            )
            self.assertEqual(
                set(calls_recorder.called_args),
                set([
                    ('A', 'D'),
                    ('A', 'E'),
                    ('A', 'F'),
                    ('B', 'D'),
                    ('B', 'E'),
                    ('B', 'F'),
                    ('C', 'D'),
                    ('C', 'E'),
                    ('C', 'F'),
                    ('Z', 'Z'),
                    (0, 10),
                    (0, 20),
                    (1, 10),
                    (1, 20),
                ])
            )


    def test_a(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                [['A', 'B', 'C'], ['D', 'E'],]
            )
            self.assertEqual(
                set(calls_recorder.called_args),
                set([
                    ('A', 'D'),
                    ('A', 'E'),
                    ('B', 'D'),
                    ('B', 'E'),
                    ('C', 'D'),
                    ('C', 'E'),
                ])
            )


    def test_1(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                [['A','B','C'], ['D','E','F']],
                [['Z'], ['Z']],
                [[0,1], [10,20]],
                shuffle_tasks=False,
                sequential_execution=True,
            )
            self.assertEqual(
                list(calls_recorder.called_args),
                [
                    ('A', 'D'),
                    ('A', 'E'),
                    ('A', 'F'),
                    ('B', 'D'),
                    ('B', 'E'),
                    ('B', 'F'),
                    ('C', 'D'),
                    ('C', 'E'),
                    ('C', 'F'),
                    ('Z', 'Z'),
                    (0, 10),
                    (0, 20),
                    (1, 10),
                    (1, 20),
                ]
            )

    def test_2(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                [['A','B','C'], ['D','E','F']],
                [['Z'], ['Z']],
                [[0,1], [10,20]],
                shuffle_tasks=True,
            )
            self.assertEqual(
                set(calls_recorder.called_args),
                set([
                    ('A', 'D'),
                    ('A', 'E'),
                    ('A', 'F'),
                    ('B', 'D'),
                    ('B', 'E'),
                    ('B', 'F'),
                    ('C', 'D'),
                    ('C', 'E'),
                    ('C', 'F'),
                    ('Z', 'Z'),
                    (0, 10),
                    (0, 20),
                    (1, 10),
                    (1, 20),
                ])
            )

    def test_3(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                [['A','B','C'], ['D','E','F']],
                [['Z'], ['Z']],
                [[0,1], [10,20]],
                chunksize=5,
            )
            self.assertEqual(
                set(calls_recorder.called_args),
                set([
                    ('A', 'D'),
                    ('A', 'E'),
                    ('A', 'F'),
                    ('B', 'D'),
                    ('B', 'E'),
                    ('B', 'F'),
                    ('C', 'D'),
                    ('C', 'E'),
                    ('C', 'F'),
                    ('Z', 'Z'),
                    (0, 10),
                    (0, 20),
                    (1, 10),
                    (1, 20),
                ])
            )

    def test_4(self):
        
        inputs = [
            {'x': ['A', 'B', 'C'],
             'y': ['D', 'E'],},
            {'x': ['A', 'B', 'C'],
             'y': 'FF'},
        ]

        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                inputs=inputs,
            )
            def _to_sorted_list(X):
                return sorted(X, key=lambda x: (x['x'], x['y']))
            self.assertEqual(
                _to_sorted_list(list(calls_recorder.called_args)),
                _to_sorted_list([
                    dict(x='A', y='D'),
                    dict(x='B', y='D'),
                    dict(x='C', y='D'),
                    dict(x='A', y='E'),
                    dict(x='B', y='E'),
                    dict(x='C', y='E'),
                    dict(x='A', y='FF'),
                    dict(x='B', y='FF'),
                    dict(x='C', y='FF'),
                ])
            )


    def test_chunked_with_ComputeAndStore(self):

        from phfnbutils.store import Hdf5StoreResultsAccessor, ComputeAndStore

        with tempfile.TemporaryDirectory() as temp_dir_name:
            storefn = os.path.join(temp_dir_name, 'tempstore.hdf5')

            compute_something_and_store = ComputeAndStore(fn_helper, storefn)

            parallel_apply_func_on_input_combinations(
                compute_something_and_store,
                [ [1,2,3,4,5], [7,8] ],
                chunksize=8,
            )

            with Hdf5StoreResultsAccessor(storefn) as store:
                self.assertEqual( set([ (r['a'], r['res_value'], r['factor'])
                                        for r in store.iterate_results() ]) ,
                                  set([ (7,  7, 1),
                                        (7, 14, 2),
                                        (7, 21, 3),
                                        (7, 28, 4),
                                        (7, 35, 5),
                                        (8,  8, 1),
                                        (8, 16, 2),
                                        (8, 24, 3),
                                        (8, 32, 4),
                                        (8, 40, 5), ]) )



class Test_get_inputs_iterable(unittest.TestCase):
    def test_args_list(self):
        total_n, iterable = get_inputs_iterable(args_values_lists=[[
                [1, 2, 3],
                ['A'],
                ['alpha', 'beta'],
            ]])
        self.assertEqual(
            list(iterable),
            [
                (1, 'A', 'alpha'),
                (1, 'A', 'beta'),
                (2, 'A', 'alpha'),
                (2, 'A', 'beta'),
                (3, 'A', 'alpha'),
                (3, 'A', 'beta'),
            ]
        )
        self.assertEqual(total_n, 6)

    def test_args_list_2(self):
        total_n, iterable = get_inputs_iterable(args_values_lists=[[
                [1, 2, 3],
                ['A'],
                ['alpha', 'beta'],
            ]], shuffle_tasks=True)
        self.assertEqual(
            set(list(iterable)),
            set([
                (1, 'A', 'alpha'),
                (1, 'A', 'beta'),
                (2, 'A', 'alpha'),
                (2, 'A', 'beta'),
                (3, 'A', 'alpha'),
                (3, 'A', 'beta'),
            ])
        )
        self.assertEqual(total_n, 6)

    def test_dicts_1(self):
        total_n, iterable = get_inputs_iterable(inputs=[
            {
                'x': [1, 2, 3],
                'y': ['A'],
                'z': ['alpha', 'beta'],
            },
        ])
        self.assertEqual(
            list(iterable),
            [
                dict(x=1, y='A', z='alpha'),
                dict(x=1, y='A', z='beta'),
                dict(x=2, y='A', z='alpha'),
                dict(x=2, y='A', z='beta'),
                dict(x=3, y='A', z='alpha'),
                dict(x=3, y='A', z='beta'),
            ]
        )
        self.assertEqual(total_n, 6)

    def test_dicts_2(self):
        total_n, iterable = get_inputs_iterable(inputs=[
            {
                'x': [1, 2, 3],
                'y': ['A'],
                'z': ['alpha', 'beta'],
            },
            {
                'x': 'x',
                'y': 'y',
                'z': 'z',
            },
            {
                'x': [1, 2, 3],
                'y': ['Z'],
                'z': 'gamma',
            },
        ])
        self.assertEqual(
            list(iterable),
            [
                dict(x=1, y='A', z='alpha'),
                dict(x=1, y='A', z='beta'),
                dict(x=2, y='A', z='alpha'),
                dict(x=2, y='A', z='beta'),
                dict(x=3, y='A', z='alpha'),
                dict(x=3, y='A', z='beta'),
                dict(x='x', y='y', z='z'),
                dict(x=1, y='Z', z='gamma'),
                dict(x=2, y='Z', z='gamma'),
                dict(x=3, y='Z', z='gamma'),
            ]
        )
        self.assertEqual(total_n, 10)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
