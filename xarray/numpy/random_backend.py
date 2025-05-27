from typing import Union, Optional, Tuple, Any
import numpy as np

__all__ = [
    "random_number_generator",
    "random_discrete_uniform",
    "random_uniform",
    "random_exponential",
    "random_normal",
    "random_geometric",
    "random_permutation"
]

def random_number_generator(
    seed : Optional[int] = None,
    *,
    device : Optional[Any] = None
) -> np.random.Generator:
    return np.random.default_rng(seed)

def random_discrete_uniform(
    shape : Union[int, Tuple[int, ...]], 
    from_num : int, 
    to_num : int, 
    /,
    *,
    rng : np.random.Generator, 
    dtype : Optional[np.dtype] = None, 
    device : None = None
) -> Tuple[np.random.Generator, np.ndarray]:
    t = rng.integers(int(from_num), int(to_num), size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

def random_uniform(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : np.random.Generator, 
    low : float = 0.0, high : float = 1.0,
    dtype : Optional[np.dtype] = None, 
    device : Optional[Any] = None
) -> Tuple[np.random.Generator, np.ndarray]:
    t = rng.uniform(float(low), float(high), size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

def random_exponential(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    rng : np.random.Generator,  
    lambd : float = 1.0, 
    dtype : Optional[np.dtype] = None, 
    device : Optional[Any] = None
) -> Tuple[np.random.Generator, np.ndarray]:
    t = rng.exponential(1.0 / float(lambd), size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

@classmethod
def random_normal(
    shape: Union[int, Tuple[int, ...]],
    /,
    *, 
    rng : np.random.Generator,
    mean : float = 0.0, std : float = 1.0, 
    dtype : Optional[np.dtype] = None, 
    device : Optional[Any] = None
) -> Tuple[np.random.Generator, np.ndarray]:
    t = rng.normal(mean, std, size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t


def random_geometric(
    shape: Union[int, Tuple[int, ...]], 
    /,
    *,
    p: float, 
    rng: np.random.Generator,
    dtype: Optional[np.dtype] = None, 
    device: Optional[Any] = None
) -> Tuple[np.random.Generator | np.ndarray]:
    t = rng.geometric(p, size=shape)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t

def random_permutation(
    n : int,
    /,
    *,
    rng: np.random.Generator,
    dtype: Optional[np.dtype] = None,
    device: Optional[Any] = None
) -> Tuple[np.random.Generator, np.ndarray]:
    t = rng.permutation(n)
    if dtype is not None:
        t = t.astype(dtype)
    return rng, t