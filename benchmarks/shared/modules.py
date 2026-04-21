"""Factory for single-signature extractor modules.

All six benchmarks wrap exactly one signature with either dspy.Predict or
dspy.ChainOfThought selected at construction time. ``build_extractor_class``
produces a dspy.Module subclass with that shape.
"""

from typing import Literal, get_args

import dspy

ModuleType = Literal["predict", "cot"]
_VALID_MODULE_TYPES = get_args(ModuleType)


def build_extractor_class(signature_cls, *, class_name: str | None = None):
    """Return a dspy.Module subclass that wraps ``signature_cls``.

    The returned class:
      - accepts ``module_type: ModuleType`` in __init__
      - exposes ``self.predict`` (the Predict/ChainOfThought instance),
        which external code (KNN demo-swap, logging) relies on
      - delegates kwargs in ``forward()`` straight through to the predictor

    Args:
        signature_cls: A ``dspy.Signature`` subclass.
        class_name:    Name for the returned class (affects repr/logs only).
                       Defaults to ``<SignatureName>Module``.
    """

    class Extractor(dspy.Module):
        VALID_MODULE_TYPES = _VALID_MODULE_TYPES

        def __init__(self, module_type: ModuleType = "predict"):
            super().__init__()
            if module_type not in _VALID_MODULE_TYPES:
                raise ValueError(
                    f"module_type must be one of {_VALID_MODULE_TYPES}, got {module_type!r}"
                )
            if module_type == "cot":
                self.predict = dspy.ChainOfThought(signature_cls)
            else:
                self.predict = dspy.Predict(signature_cls)

        def forward(self, **kwargs):
            return self.predict(**kwargs)

    Extractor.__name__ = class_name or f"{signature_cls.__name__}Module"
    Extractor.__qualname__ = Extractor.__name__
    return Extractor
