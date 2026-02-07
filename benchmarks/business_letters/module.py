"""DSPy Module wrapping the business letter extraction signature."""

import dspy

from benchmarks.business_letters.signature import BusinessLetterExtraction


class BusinessLetterExtractor(dspy.Module):
    VALID_MODULE_TYPES = ("predict", "cot")

    def __init__(self, module_type: str = "predict"):
        super().__init__()
        if module_type not in self.VALID_MODULE_TYPES:
            raise ValueError(f"module_type must be one of {self.VALID_MODULE_TYPES}, got {module_type!r}")
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(BusinessLetterExtraction)
        else:
            self.predict = dspy.Predict(BusinessLetterExtraction)

    def forward(self, page_images):
        return self.predict(page_images=page_images)


# Canonical alias for script compatibility
Extractor = BusinessLetterExtractor
