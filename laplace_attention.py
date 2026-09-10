"""Compatibility shim: the layer lives in lapa/layer.py (package `lapa`)."""
from lapa.layer import *  # noqa: F401,F403
from lapa.layer import LaplaceAttention, LaplaceConfig, LongHead, ShortHead  # noqa: F401
