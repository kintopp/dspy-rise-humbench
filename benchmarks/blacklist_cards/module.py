"""DSPy Module wrapping the blacklist card extraction signature.

Defined as a concrete subclass (not via ``build_extractor_class``) so that
``dspy.Refine`` can introspect the class source via ``inspect.getsource`` —
the factory pattern breaks under dspy>=3.1 because Refine caches
``inspect.getsource(module.__class__)`` at construction and the factory's
dynamically-created class has no source file.
"""

from typing import Literal, get_args

import dspy

from benchmarks.blacklist_cards.signature import BlacklistCardExtraction

ModuleType = Literal["predict", "cot"]
_VALID_MODULE_TYPES = get_args(ModuleType)


class Extractor(dspy.Module):
    VALID_MODULE_TYPES = _VALID_MODULE_TYPES

    def __init__(self, module_type: ModuleType = "predict"):
        super().__init__()
        if module_type not in _VALID_MODULE_TYPES:
            raise ValueError(
                f"module_type must be one of {_VALID_MODULE_TYPES}, got {module_type!r}"
            )
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(BlacklistCardExtraction)
        else:
            self.predict = dspy.Predict(BlacklistCardExtraction)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
