from typing import Protocol, TypeVar, Optional, Any, Tuple, Union, Type, TypedDict, List
from abc import abstractmethod
from array_api_typing.typing_compat import ArrayAPINamespaceArrayT

class AtResult(Protocol[ArrayAPINamespaceArrayT]):
    @abstractmethod
    def set(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ):
        pass

    @abstractmethod
    def add(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass

    @abstractmethod
    def subtract(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass

    @abstractmethod
    def multiply(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass

    @abstractmethod
    def divide(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass

    @abstractmethod
    def power(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass

    @abstractmethod
    def min(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass
    
    @abstractmethod
    def max(
        self,
        y: Union[ArrayAPINamespaceArrayT, float, int, complex],
        /,
        copy: Optional[bool] = None,
    ) -> ArrayAPINamespaceArrayT:
        pass