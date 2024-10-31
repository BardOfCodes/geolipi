from typing import List, Tuple
class Settings:

    COORD_MODE: str = "centered" # or "bound"
    ROT_ORDER: Tuple[str] = ("X", "Y", "Z")
    COLOR_CLAMP: bool = True


def update_settings(coord_mode: str = "centered", 
                    rot_order: Tuple[str] = ("X", "Y", "Z"),
                    color_clamp: bool = True):
    Settings.COORD_MODE = coord_mode
    Settings.ROT_ORDER = rot_order
    Settings.COLOR_CLAMP = color_clamp
