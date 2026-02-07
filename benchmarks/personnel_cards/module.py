"""DSPy Module wrapping the personnel card extraction signature."""

import dspy

from benchmarks.personnel_cards.signature import PersonnelCardExtraction


class PersonnelCardExtractor(dspy.Module):
    VALID_MODULE_TYPES = ("predict", "cot")

    def __init__(self, module_type: str = "predict"):
        super().__init__()
        if module_type not in self.VALID_MODULE_TYPES:
            raise ValueError(f"module_type must be one of {self.VALID_MODULE_TYPES}, got {module_type!r}")
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(PersonnelCardExtraction)
        else:
            self.predict = dspy.Predict(PersonnelCardExtraction)

    def forward(self, card_image):
        return self.predict(card_image=card_image)


# Canonical alias for script compatibility
Extractor = PersonnelCardExtractor
