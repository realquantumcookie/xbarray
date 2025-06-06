from typing import Any, Union, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
from xbarray.base import ComputeBackend, SupportsDLPack

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
    return isinstance(data, jax.Array)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    return jax.numpy.asarray(data, dtype=dtype, device=device)

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    data_dlpack = other_backend.to_dlpack(data)
    return jax.dlpack.from_dlpack(data_dlpack)
    # except Exception as e:
    #     # jax sometimes has tiling issues with dlpack converted data
    #     np = other_backend.to_numpy(data)
    #     return from_numpy(np)

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    if data.dtype == jax.dtypes.bfloat16:
        data = data.astype(np.float32)
    return np.asarray(data)

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
    return dtype == jax.dtypes.bfloat16 or np.issubdtype(dtype, np.floating)

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == np.bool_ or dtype == bool

from xbarray.common.implementations import *
if hasattr(jax.numpy, "__array_api_version__"):
    compat_module = jax.numpy
else:
    import jax.experimental.array_api as compat_module
abbreviate_array = get_abbreviate_array_function(
    backend=compat_module,
    default_integer_dtype=default_integer_dtype,
    func_dtype_is_real_floating=dtype_is_real_floating,
    func_dtype_is_real_integer=dtype_is_real_integer,
    func_dtype_is_boolean=dtype_is_boolean
)
def map_fn_over_arrays(
    data : Any, func : Callable[[ARRAY_TYPE], ARRAY_TYPE]
):
    return jax.tree.map(
        func,
        data
    )
