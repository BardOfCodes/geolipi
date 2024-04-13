from typing import List, Tuple
class Settings:

    COORD_MODE: str = "bound" # or "centered"
    ROT_ORDER: Tuple[str] = ("X", "Y", "Z")


def update_settings(coord_mode: str = "bound", rot_order: Tuple[str] = ("X", "Y", "Z")):
    Settings.COORD_MODE = coord_mode
    Settings.ROT_ORDER = rot_order
