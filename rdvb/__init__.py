"""Recurrent-depth vision backbone with per-token adaptive compute."""

from .model import RDVB, RDVBConfig, PlainViT, count_parameters

__all__ = ["RDVB", "RDVBConfig", "PlainViT", "count_parameters"]
