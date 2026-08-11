"""Vendor-neutral suction/aspiration controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SuctionController(ABC):
    """Actuate the configured suction or aspiration device through one small contract."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def on(self) -> None: ...

    @abstractmethod
    def off(self) -> None: ...

    @abstractmethod
    def is_on(self) -> bool:
        """Return True only when vacuum output/feedback is valid for holding an object."""
        ...

    @abstractmethod
    def close(self) -> None: ...
