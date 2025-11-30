from .config_base import VRConfig
from .fiqa import fiqa
from .fever import fever
from .scifact import scifact
from .enronmail import enronmail
from .nfcorpus import nfcorpus
from .twcs import twcs
from .arguana import arguana
from .chatdoctor import chatdoctor

__all__ = [
    "VRConfig", 
    "fiqa",
    "fever",
    "scifact",
    "enronmail",
    "nfcorpus",
    "twcs",
    "arguana",
    "chatdoctor"
    ]