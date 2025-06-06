from typing import Any, Union, Optional
import numpy as np
import torch
from ._typing import ARRAY_TYPE, DTYPE_TYPE, DEVICE_TYPE, RNG_TYPE
from xbarray.base import ComputeBackend, SupportsDLPack

PYTORCH_DTYPE_CAST_MAP = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
    torch.float8_e4m3fn: torch.float16,
    torch.float8_e5m2: torch.float16,
}

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

default_integer_dtype = torch.int32
default_floating_dtype = torch.float32
default_boolean_dtype = torch.bool

def is_backendarray(data : Any) -> bool:
    return isinstance(data, torch.Tensor)

def from_numpy(
    data : np.ndarray,
    /,
    *,
    dtype : Optional[DTYPE_TYPE] = None,
    device : Optional[DEVICE_TYPE] = None
) -> ARRAY_TYPE:
    t = torch.from_numpy(data)
    target_dtype = dtype if dtype is not None else PYTORCH_DTYPE_CAST_MAP.get(t.dtype, t.dtype)
    if target_dtype is not None or device is not None:
        t = t.to(device=device, dtype=target_dtype)
    return t

def from_other_backend(
    other_backend: ComputeBackend,
    data: Any,
    /,
) -> ARRAY_TYPE:
    dat_dlpack = other_backend.to_dlpack(data)
    return torch.from_dlpack(dat_dlpack)

def to_numpy(
    data : ARRAY_TYPE
) -> np.ndarray:
    # Torch bfloat16 is not supported by numpy
    if data.dtype == torch.bfloat16:
        data = data.to(torch.float32)
    return data.cpu().numpy()

def to_dlpack(
    self,
    data: ARRAY_TYPE,
    /,
) -> SupportsDLPack:
    return data

def dtype_is_real_integer(
    dtype: DTYPE_TYPE
) -> bool:
    # https://pytorch.org/docs/stable/tensors.html#id12
    return dtype in [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, 
        torch.int,
        torch.long
    ]

def dtype_is_real_floating(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype in [
        torch.float16, torch.float32, torch.float64, 
        torch.float, torch.double, 
        torch.bfloat16
    ]

def dtype_is_boolean(
    dtype: DTYPE_TYPE
) -> bool:
    return dtype == torch.bool

from xbarray.common.implementations import get_abbreviate_array_function, get_map_fn_over_arrays_function
from array_api_compat import torch as compat_module
abbreviate_array = get_abbreviate_array_function(
    compat_module, 
    default_integer_dtype=default_integer_dtype, 
    func_dtype_is_real_floating=dtype_is_real_floating,
    func_dtype_is_real_integer=dtype_is_real_integer,
    func_dtype_is_boolean=dtype_is_boolean
)
map_fn_over_arrays = get_map_fn_over_arrays_function(
    is_backendarray=is_backendarray,
)