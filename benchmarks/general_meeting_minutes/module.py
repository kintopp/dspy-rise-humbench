"""DSPy Module wrapping the General Meeting Minutes extraction signature.

Concrete class (not factory) so dspy.Refine can introspect the source.
"""

from typing import Literal, get_args

import dspy

from benchmarks.general_meeting_minutes.signature import MinutesExtraction

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
            self.predict = dspy.ChainOfThought(MinutesExtraction)
        else:
            self.predict = dspy.Predict(MinutesExtraction)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
