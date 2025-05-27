import functools
from typing import Any, Union, Optional
import numpy as np
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
from xarray.base import ComputeBackend, SupportsDLPack

__all__ = [
    "default_integer_dtype",
    "default_floating_dtype",
    "default_boolean_dtype",
    "is_backendarray",
    "from_numpy",
    "from_other_backend",
    "to_numpy",
    "to_dlpack",
    "dtype_is_real_integer",
    "dtype_is_real_floating",
    "dtype_is_boolean",
    "abbreviate_array",
]

default_integer_dtype = int
default_floating_dtype = float
default_boolean_dtype = bool

def is_backendarray(data : Any) -> bool:
    return isinstance(data, np.ndarray)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    return data

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    return other_backend.to_numpy(data)

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    return data

def to_dlpack(
    self,
    data: ARRAY_TYPE,
    /,
) -> SupportsDLPack:
    return data

def dtype_is_real_integer(
    dtype: DTYPE_TYPE
) -> bool:
    return np.issubdtype(dtype, np.integer)

def dtype_is_real_floating(
    dtype: DTYPE_TYPE
) -> bool:
    return np.issubdtype(dtype, np.floating)

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == np.bool_ or dtype == bool

from xarray.common.implementations import abbreviate_array as abbreviate_array_common
from array_api_compat import numpy as np_compat
abbreviate_array = functools.partial(
    abbreviate_array_common,
    np_compat
)