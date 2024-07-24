from enum import IntEnum

class BoneType(IntEnum):
    """
    @brief Defines the type a bone can have
    """
    NONE = 0
    HEAD = 1
    BODY = 2
    LIMB = 3
    WING = 4
    MIXED = 5

class JointType(IntEnum):
    """
    @brief Defines the type a joint can have
    """
    NONE = 0
    HEAD = 1
    BODY = 2
    LIMB = 3
    WING = 4
    MIXED = 5 # mixed joints connect bones of different types