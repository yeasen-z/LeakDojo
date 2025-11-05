from .config_base import VRConfig
from .fiqa import fiqa
from .fever import fever_filtered, fever_filtered_overlap

__all__ = [
    "VRConfig", 
    "fiqa",
    "fever_filtered",
    "fever_filtered_overlap"
    ]