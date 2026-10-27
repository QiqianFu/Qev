"""Qev: Qwen-based decision models with candidate branches."""
__version__ = "0.1.0"
__all__ = ["Qev", "QevSession", "__version__"]


def __getattr__(name):
    if name == 'Qev':
        from .api import Qev
        return Qev
    if name == 'QevSession':
        from .session import QevSession
        return QevSession
    raise AttributeError(name)
