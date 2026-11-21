"""
src/utils.py - Lightweight Data Sanitization and Response Utilities.

Provides recursive sanitization for NaN, +inf, and -inf floating-point values
to ensure strict compliance with the RFC 8259 JSON standard.
"""

import math
import typing
from typing import Any
import numpy as np
import pandas as pd
from starlette.responses import JSONResponse


def clean_nan(obj: Any) -> Any:
    """
    Recursively converts float NaN, +inf, and -inf values to None to produce
    RFC 8259-compliant JSON payloads without serialization errors.

    Handles:
    - Standard Python floats, ints, booleans, strings, bytes, and None.
    - Python collections: dict, list, tuple, and set.
    - NumPy scalars: numpy.floating, numpy.integer, numpy.bool_.
    - NumPy arrays: numpy.ndarray.
    - Pandas containers and missing value sentinels: pd.DataFrame, pd.Series, pd.NA, pd.NaT.
    """
    if obj is None:
        return None

    # 1. Containers and Sequences
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clean_nan(v) for v in obj)
    if isinstance(obj, set):
        return [clean_nan(v) for v in obj]

    # 2. Pandas & NumPy Containers
    if isinstance(obj, pd.DataFrame):
        return clean_nan(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return [clean_nan(v) for v in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [clean_nan(v) for v in obj.tolist()]

    # 3. Booleans (MUST check before int because bool is a subclass of int in Python)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    # 4. Integers & NumPy Integer Scalars
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # 5. Floats & NumPy Floating Scalars
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    # 6. Strings and Bytes
    if isinstance(obj, (str, bytes)):
        return obj

    # 7. Pandas Missing Values (pd.NA, pd.NaT, pd.isna)
    if obj is pd.NA or obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass

    return obj


class SafeJSONResponse(JSONResponse):
    """
    Custom Starlette/FastAPI JSONResponse that guarantees JSON standard compliance
    by recursively sanitizing NaN, +inf, and -inf values to None before serialization.
    """

    def render(self, content: typing.Any) -> bytes:
        return super().render(clean_nan(content))
