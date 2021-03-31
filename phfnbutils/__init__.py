# phfnbutils

# This determines the package version number.
__version__ = "0.2.0"


from ._general_utils import (
    TimeThis, TimeThisResult,
)

#from ._mp_utils import (
#    parallel_apply_func_on_input_combinations,
#)

# lazy import of mp module, keeps old code running
def parallel_apply_func_on_input_combinations(*args, **kwargs):
    """
    Obsolete, please use
    :py:func:`phfnbutils.mp.parallel_apply_func_on_input_combinations()`
    instead.
    """
    from . import mp
    mp.parallel_apply_func_on_input_combinations(*args, **kwargs)
