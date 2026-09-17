"""Canary - minimal, modular, self-improving personal agent harness.

Top-level imports are lazy so that importing a submodule (for example
``canary.core.memory``) never drags in the whole agent. ``from canary import
Agent`` keeps working once the implementation module exists.
"""

from canary.core.config import Config

__version__ = "0.1.0"

__all__ = ["Agent", "Config", "__version__"]


def __getattr__(name: str):
    if name == "Agent":
        from canary.core.agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
