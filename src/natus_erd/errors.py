"""Public exceptions raised by :mod:`natus_erd`."""


class NatusERDError(Exception):
    """Base class for reader errors."""


class UnsupportedFormatError(NatusERDError):
    """The recording uses a schema or hardware layout not implemented here."""


class DataIntegrityError(NatusERDError):
    """A NeuroWorks file is truncated, inconsistent, or otherwise corrupt."""


class ResourceLimitError(NatusERDError):
    """A request or file exceeds an explicit reader resource budget.

    This does not necessarily mean the file is corrupt. Use smaller windows,
    or explicitly choose appropriate :class:`ReadLimits` for trusted data.
    """
