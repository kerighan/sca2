"""
Version v0 -- the frozen reference.

Semantics contract for every later version, and the slow-but-obvious
implementation of it. Defined in sca2/ref.py; this module only exposes it under
the version interface so `SCA2_VERSION=v0` works as a drop-in.

`python -m sca2.test_fidelity` proves it reproduces the original
bench_tinypython.py exactly (prefill bit-identical, decode to 1.8e-15).
"""
from ..ref import CHeadRef, DHeadRef, SCA2Layer

CHead = CHeadRef
DHead = DHeadRef
Layer = SCA2Layer
STATUS = "reference"
NOTE = "frozen reference (contract)"
