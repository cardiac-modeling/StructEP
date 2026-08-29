"""Exception hierarchy for StructEP inference."""


class StructEPError(RuntimeError):
    """Base class for user-facing StructEP errors."""


class RegistryError(StructEPError):
    """Raised when model metadata or checkpoint layout is invalid."""


class InputBatchError(StructEPError):
    """Raised when a model-ready inference batch violates the input schema."""


class InferenceError(StructEPError):
    """Raised when a model cannot produce a valid prediction."""
