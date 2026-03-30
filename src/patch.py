"""
patch.py — Monkey-patch nexaai ProfileData on Windows.

On Windows, nexa_bridge.dll returns garbage bytes in the stop_reason field
of profiling data after VLM generation. This causes a UnicodeDecodeError in
ProfileData.from_c_struct(). The generated text is unaffected.

Import this module before any other nexaai import.
"""

from nexaai.nexa_sdk import types as _nexa_types

_orig = _nexa_types.ProfileData.from_c_struct.__func__


@classmethod
def _safe_from_c_struct(cls, c_struct):
    try:
        return _orig(cls, c_struct)
    except (UnicodeDecodeError, AttributeError):
        return cls(stop_reason="unknown")


_nexa_types.ProfileData.from_c_struct = _safe_from_c_struct
