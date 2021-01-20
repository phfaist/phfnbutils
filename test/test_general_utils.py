import unittest
import tempfile
import os.path

import multiprocessing

import h5py

import logging

import numpy as np

from phfnbutils import parallel_apply_func_on_input_combinations


import multiprocessing


# ------------------------------------------------------------------------------



class CallsRecorder:
    def __init__(self, initial_list=None):
        if initial_list is not None:
            self.called_args = initial_list
        else:
            self.called_args = []

    def __call__(self, arg):
        self.called_args.append(arg)


class Test_parallel_apply_func_on_input_combinations(unittest.TestCase):

    def test_0(self):
        
        with multiprocessing.Manager() as manager:
            calls_recorder = CallsRecorder(manager.list())
            parallel_apply_func_on_input_combinations(
                calls_recorder,
                ['ABC', 'DEF'],
                ['Z', 'Z'],
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
                ['ABC', 'DEF'],
                ['Z', 'Z'],
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
                ['ABC', 'DEF'],
                ['Z', 'Z'],
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
                ['ABC', 'DEF'],
                ['Z', 'Z'],
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
