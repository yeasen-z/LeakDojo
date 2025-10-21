"""
IKEA-style query generator adapted to this project's framework.

Key changes vs upstream:
- Aligns to src.interfaces.QueryGenerator and LLMManager (string prompt I/O)
- Uses src.utils.get_embed_model (LangChain HuggingFaceEmbeddings)
- Removes undefined dependencies (MutationAttacker, gpt_generator, token counters, pandas I/O)
- Replaces sentence-transformers encode() calls with embed_documents/embed_query
- Provides a simple generate() that returns a list of questions
"""

import json
import random
import re
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from src.interfaces import LLMManager, QueryGenerator
from src.utils import get_embed_model


class IKEAQueryGenerator(QueryGenerator):
    """A lightweight IKEA-style QueryGenerator.

    Responsibilities:
    - Manage an anchor-word pool via LLM
    - Turn anchor words into broad questions compatible with the pipeline
    - Provide optional utilities for similarity filtering
    """

    def __init__(
        self
    ):
        pass