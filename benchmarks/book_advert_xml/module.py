"""DSPy Module wrapping the Book Advert XML correction signature.

Concrete subclass (not via ``build_extractor_class``) so ``dspy.Refine`` can
introspect the class source via ``inspect.getsource`` — see
benchmarks/blacklist_cards/module.py for the canonical rationale.
"""

from typing import Literal, get_args

import dspy

from benchmarks.book_advert_xml.signature import CorrectMalformedXml

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
            self.predict = dspy.ChainOfThought(CorrectMalformedXml)
        else:
            self.predict = dspy.Predict(CorrectMalformedXml)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
