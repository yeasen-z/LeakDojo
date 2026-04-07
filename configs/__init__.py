from .config_base import VRConfig
from .corpus.fiqa import fiqa
from .corpus.scifact import scifact
from .corpus.enronmail import enronmail
from .corpus.nfcorpus import nfcorpus

__all__ = [
    "VRConfig", 
    "fiqa",
    "scifact",
    "enronmail",
    "nfcorpus",
    ]