"""Compatibility shim for the historical pickle5 backport.

This package simply re-exports the standard library's :mod:`pickle` module so
that projects depending on ``pickle5`` continue to work on Python 3.8+ where
pickle protocol 5 is already available.
"""

from pickle import *  # noqa: F401,F403
import pickle as _pickle
import sys as _sys

__all__ = getattr(_pickle, "__all__", [])

# Ensure ``import pickle5._pickle`` resolves to the stdlib implementation.
_sys.modules.setdefault("pickle5._pickle", _pickle)
