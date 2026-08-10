"""Vendor-neutral electric suction controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SuctionController(ABC):
    """Actuate vacuum only through this intentionally small contract."""

    @abstractmethod
    def on(self) -> None: ...

    @abstractmethod
    def off(self) -> None: ...

    @abstractmethod
    def is_on(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...
