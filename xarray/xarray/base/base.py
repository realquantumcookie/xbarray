from typing import Optional, Generic, TypeVar, Dict, Union, Any, Sequence, SupportsFloat, Tuple, Type, Callable, Mapping, Protocol
import abc
from array_api_typing.typing_extra import *
import array_api_compat
import numpy as np

BArrayType = TypeVar("BArrayType", covariant=True, bound=ArrayAPIArray)
BDeviceType = TypeVar("BDeviceType", covariant=True, bound=ArrayAPIDevice)
BDtypeType = TypeVar("BDtypeType", covariant=True, bound=ArrayAPIDType)
BRNGType = TypeVar("BRNGType", covariant=True)
class RNGBackend(Protocol[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    @abc.abstractmethod
    def random_number_generator(
        self, 
        seed : Optional[int] = None, 
        *,
        device : Optional[BDeviceType] = None
    ) -> BRNGType:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_discrete_uniform(
        self, 
        shape : Union[int, Tuple[int, ...]], 
        from_num : int, 
        to_num : Optional[int], 
        /,
        *,
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None, 
    ) -> Tuple[BRNGType, BArrayType]:
        """
        Sample from a discrete uniform distribution [from_num, to_num) with shape `shape`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_uniform(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        low : float = 0.0, high : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_exponential(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        lambd : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None, 
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_normal(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *,
        rng : BRNGType, 
        mean : float = 0.0, std : float = 1.0, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_geometric(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        /,
        *, 
        p : float, 
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None,
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

    @abc.abstractmethod
    def random_permutation(
        self, 
        n : int, 
        /,
        *, 
        rng : BRNGType, 
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None
    ) -> Tuple[BRNGType, BArrayType]:
        raise NotImplementedError

class ComputeBackend(Protocol[BArrayType, BDeviceType, BDtypeType, BRNGType], ArrayAPINamespace[BArrayType, BDeviceType, BDtypeType]):
    default_integer_dtype : BDtypeType
    default_floating_dtype : BDtypeType
    default_boolean_dtype : BDtypeType
    random : RNGBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]

    @abc.abstractmethod
    def is_backendarray(self, data : Any) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def from_numpy(
        self, 
        data : np.ndarray, 
        /,
        *,
        dtype : Optional[BDtypeType] = None, 
        device : Optional[BDeviceType] = None
    ) -> BArrayType:
        raise NotImplementedError

    @abc.abstractmethod
    def to_numpy(
        self, 
        data : BArrayType
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dlpack(
        self, 
        data : BArrayType, 
        /,
    ) -> SupportsDLPack:
        raise NotImplementedError

    @abc.abstractmethod
    def dtype_is_real_integer(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def dtype_is_real_floating(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def dtype_is_boolean(self, dtype : BDtypeType) -> bool:
        raise NotImplementedError
    
    # @abc.abstractmethod
    # def abbreviate_array(cls, array : BArrayType, try_cast_scalar : bool = True) -> Union[float, int, BArrayType]:
    #     """
    #     Abbreivates an array to a single element if possible.
    #     Or, if some dimensions are the same, abbreviates to a smaller array (but with the same number of dimensions).
    #     """
    #     abbr_array = array
    #     idx = cls.array_api_namespace.zeros(1, dtype=cls.default_integer_dtype, device=cls.get_device(abbr_array))
    #     for dim_i in range(len(array.shape)):
    #         first_elem = cls.array_api_namespace.take(abbr_array, idx, axis=dim_i)
    #         if cls.array_api_namespace.all(abbr_array == first_elem):
    #             abbr_array = first_elem
    #         else:
    #             continue
    #     if try_cast_scalar:
    #         if all(i == 1 for i in abbr_array.shape):
    #             elem = abbr_array[tuple([0] * len(abbr_array.shape))]
    #             if cls.dtype_is_real_floating(elem.dtype):
    #                 return float(elem)
    #             elif cls.dtype_is_real_integer(elem.dtype):
    #                 return int(elem)
    #             elif cls.dtype_is_boolean(elem.dtype):
    #                 return bool(elem)
    #             else:
    #                 raise ValueError(f"Abbreviated array element dtype must be a real floating or integer or boolean type, actual dtype: {elem.dtype}")
    #     else:
    #         return array
    
    # @classmethod
    # def map_fn_over_arrays(cls, data : Any, func : Callable[[BArrayType], BArrayType]) -> Any:
    #     """
    #     Map a function to the data.
    #     """
    #     if cls.is_backendarray(data):
    #         return func(data)
    #     elif isinstance(data, Mapping):
    #         return {k: cls.map_fn_over_arrays(v, func) for k, v in data.items()}
    #     elif isinstance(data, tuple):
    #         return tuple(cls.map_fn_over_arrays(i, func) for i in data)
    #     elif isinstance(data, Sequence):
    #         return [cls.map_fn_over_arrays(i, func) for i in data]
    #     else:
    #         return data