"""
A custom linear algebra module with a few machine learning tools
built entirely from scratch using pure Python.
"""

from .vector import Vector
from .matrix import Matrix
from .tools import svd, pinv, pca

__all__ = [
    "Vector",
    "Matrix",
    "svd",
    "pinv",
    "pca"
]

__version__ = "1.0"
__author__ = "Chinmoy Majumder"
