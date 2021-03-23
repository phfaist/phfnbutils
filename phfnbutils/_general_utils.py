import os
import os.path
import sys

import logging
logger = logging.getLogger(__name__)

import datetime


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

