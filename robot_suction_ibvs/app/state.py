"""System state definitions for the explicit picking state machine."""

from enum import Enum, auto


class SystemState(Enum):
    INIT = auto()
    MOVE_TO_OBSERVE = auto()
    GLOBAL_DETECT = auto()
    SELECT_TARGET = auto()
    ALIGN_IBVS = auto()
    FINAL_XY_APPROACH = auto()
    DESCEND = auto()
    SUCTION = auto()
    LIFT = auto()
    RETURN_OBSERVE = auto()
    RECOVER = auto()
    FINISHED = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()
