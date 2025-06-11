import jax.numpy

if hasattr(jax.numpy, "__array_api_version__"):
    compat_module = jax.numpy
    from jax.numpy import *
else:
    import jax.experimental.array_api as compat_module
    from jax.experimental.array_api import *

from array_api_compat.common._helpers import *

simplified_name = "jax"

# Import and bind all functions from array_api_extra before exposing them
import array_api_extra
from functools import partial
for api_name in dir(array_api_extra):
    if api_name.startswith('_'):
        continue

    if api_name == 'at':
        globals()[api_name] = getattr(array_api_extra, api_name)
    else:
        globals()[api_name] = partial(
            getattr(array_api_extra, api_name),
            xp=compat_module
        )

from ._typing import *
from ._extra import *
__import__(__package__ + ".random")