from xbarray.cls_impl.cls_base import ComputeBackendImplCls
from ._src.implementations import jax as jax_impl

class JaxComputeBackend(metaclass=ComputeBackendImplCls):
    ARRAY_TYPE = jax_impl.ARRAY_TYPE
    DTYPE_TYPE = jax_impl.DTYPE_TYPE
    DEVICE_TYPE = jax_impl.DEVICE_TYPE
    RNG_TYPE = jax_impl.RNG_TYPE

for name in dir(jax_impl):
    if not name.startswith('_') or name in [
        '__array_namespace_info__',
        '__array_api_version__',
    ]:
        setattr(JaxComputeBackend, name, getattr(jax_impl, name))

__all__ = [
    'JaxComputeBackend',
]