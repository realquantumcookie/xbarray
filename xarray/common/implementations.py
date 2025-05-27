from typing import Any, Union
from ..base import ComputeBackend, BArrayType

def abbreviate_array(backend : ComputeBackend[BArrayType, Any, Any, Any], array : BArrayType, try_cast_scalar : bool = True) -> Union[float, int, BArrayType]:
    """
    Abbreivates an array to a single element if possible.
    Or, if some dimensions are the same, abbreviates to a smaller array (but with the same number of dimensions).
    """
    abbr_array = array
    idx = backend.zeros(1, dtype=backend.default_integer_dtype, device=backend.device(abbr_array))
    for dim_i in range(len(array.shape)):
        first_elem = backend.take(abbr_array, idx, axis=dim_i)
        if backend.all(abbr_array == first_elem):
            abbr_array = first_elem
        else:
            continue
    if try_cast_scalar:
        if all(i == 1 for i in abbr_array.shape):
            elem = abbr_array[tuple([0] * len(abbr_array.shape))]
            if backend.dtype_is_real_floating(elem.dtype):
                return float(elem)
            elif backend.dtype_is_real_integer(elem.dtype):
                return int(elem)
            elif backend.dtype_is_boolean(elem.dtype):
                return bool(elem)
            else:
                raise ValueError(f"Abbreviated array element dtype must be a real floating or integer or boolean type, actual dtype: {elem.dtype}")
    else:
        return array