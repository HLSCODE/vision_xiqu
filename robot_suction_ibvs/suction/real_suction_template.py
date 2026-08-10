"""Explicit integration template for physical suction hardware."""

from __future__ import annotations

from suction.base import SuctionController


class RealSuctionController(SuctionController):
    """Implement using the confirmed GPIO, serial, or PLC protocol only."""

    def on(self) -> None:
        raise NotImplementedError("TODO: send the verified physical-vacuum ON command here")

    def off(self) -> None:
        raise NotImplementedError("TODO: send the verified physical-vacuum OFF command here")

    def is_on(self) -> bool:
        raise NotImplementedError("TODO: return actual or cached suction state here")

    def close(self) -> None:
        self.off()
