"""Keystone: post-training pipeline for foundation_llm checkpoints.

Importing this package must stay free of `torch` imports (DV-3): the CLI decides the device
and hides GPUs *before* torch is first imported.
"""
__version__ = "0.1.0"
