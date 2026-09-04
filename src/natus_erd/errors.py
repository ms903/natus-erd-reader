"""Public exceptions raised by :mod:`natus_erd`."""


class NatusERDError(Exception):
    """Base class for reader errors."""


class UnsupportedFormatError(NatusERDError):
    """The recording uses a schema or hardware layout not implemented here."""


class DataIntegrityError(NatusERDError):
    """A NeuroWorks file is truncated, inconsistent, or otherwise corrupt."""
